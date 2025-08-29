import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================
# CONFIGURABLE PARAMETERS
# ======================
# Data generation parameters
NUM_SEQUENCES = 100  # Number of simulation sequences for training
SEQ_LENGTH = 100  # Length of each simulation sequence
C_MIN = 4.0  # Minimum target y value during training
C_MAX = 8.0  # Maximum target y value during training

# Model parameters
HIDDEN_SIZE = 64  # Size of hidden layers in the neural network

# Training parameters
EPOCHS = 200  # Number of training epochs
LEARNING_RATE = 0.001  # Learning rate for optimizer
BATCH_SIZE = 256  # Batch size for training

# Testing parameters
TEST_SEQ_LENGTH = 300  # Length of test simulation
SWITCH_STEP = 200  # Time step to switch target value

# Output directory
OUTPUT_DIR = "controller_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================
# DATA GENERATION
# ======================
def generate_data(num_sequences, seq_length, c_min, c_max):
    """
    Generate training data by simulating the system with random inputs
    and calculating required x for target y values.

    Args:
        num_sequences: Number of simulation sequences
        seq_length: Length of each sequence
        c_min, c_max: Range of target y values

    Returns:
        inputs: Training inputs [y_prev, y_prev_prev, z_prev, x_prev_prev, target_y]
        targets: Required x values to achieve target_y
    """
    all_inputs = []
    all_targets = []

    for _ in range(num_sequences):
        # Initialize sequences
        y = np.zeros(seq_length)
        z = np.zeros(seq_length)
        x = np.zeros(seq_length)

        # Set random initial values
        y[0] = np.random.uniform(-1, 1)
        y[1] = np.random.uniform(-1, 1)
        z[0] = np.random.uniform(-1, 1)
        x[0] = np.random.uniform(-1, 1)
        x[1] = np.random.uniform(-1, 1)

        # Generate random inputs and simulate system
        for i in range(2, seq_length):
            x[i] = np.random.uniform(-1, 1)
            y[i] = y[i - 1] + 0.01 * y[i - 2] + 8 * x[i - 1] - 0.3 * x[i - 2] + 0.1 * z[i - 1]
            z[i] = z[i - 1] + 2 * x[i - 1] + 0.11

        # Create training samples
        for i in range(2, seq_length):
            state = [y[i - 1], y[i - 2], z[i - 1], x[i - 2]]
            target_y = np.random.uniform(c_min, c_max)

            # Calculate required x to achieve target_y
            required_x = (target_y - y[i - 1] - 0.01 * y[i - 2] + 0.3 * x[i - 2] - 0.1 * z[i - 1]) / 8

            all_inputs.append(state + [target_y])
            all_targets.append(required_x)

    return np.array(all_inputs), np.array(all_targets)


# ======================
# NEURAL NETWORK MODEL
# ======================
class Controller(nn.Module):
    """Neural network controller that predicts x to maintain constant y"""

    def __init__(self, input_size=5, hidden_size=64):
        super(Controller, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)


# ======================
# TRAINING FUNCTION
# ======================
def train_model(model, inputs, targets, epochs, lr, batch_size):
    """Train the controller model"""
    # Convert to tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    # Create dataset and loader
    dataset = torch.utils.data.TensorDataset(inputs_tensor, targets_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}')

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss.png'))
    plt.close()

    return model


# ======================
# TESTING FUNCTION
# ======================
def test_model(model_path, seq_length, switch_step):
    """Test the controller with changing target value"""
    # Load trained model
    model = Controller(input_size=5, hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Initialize sequences
    y = np.zeros(seq_length)
    z = np.zeros(seq_length)
    x = np.zeros(seq_length)

    # Initial conditions (starting from zero state)
    y[0] = 0.0
    y[1] = 0.0
    z[0] = 0.0
    x[0] = 0.0
    x[1] = 0.0

    # Simulate system with controller
    for i in range(2, seq_length):
        # Determine target value (changes at switch_step)
        target_y = 5.0 if i <= switch_step else 7.0

        # Current state: [y[i-1], y[i-2], z[i-1], x[i-2], target_y]
        state = np.array([y[i - 1], y[i - 2], z[i - 1], x[i - 2], target_y])
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # Get controller's recommendation for x[i-1]
        with torch.no_grad():
            pred_x = model(state_tensor).item()

        # Apply the recommended input
        x[i - 1] = pred_x

        # Calculate next state
        y[i] = y[i - 1] + 0.01 * y[i - 2] + 8 * x[i - 1] - 0.3 * x[i - 2] + 0.1 * z[i - 1]
        z[i] = z[i - 1] + 2 * x[i - 1] + 0.11

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y, 'b-', linewidth=2, label='Controlled y')
    plt.axvline(x=switch_step, color='k', linestyle='--', alpha=0.7, label='Target change')
    plt.axhline(y=5, color='r', linestyle='--', label='Target (y=5)')
    plt.axhline(y=7, color='g', linestyle='--', label='Target (y=7)')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('y Value', fontsize=12)
    plt.title('Controller Performance: Maintaining Constant Output', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'controlled_output.png'), bbox_inches='tight')
    plt.close()

    # Save data for analysis
    np.savez(
        os.path.join(OUTPUT_DIR, 'test_results.npz'),
        y=y, x=x, z=z, switch_step=switch_step
    )

    print(f"Test completed! Results saved in {OUTPUT_DIR}")
    print(f"Final y value: {y[-1]:.4f} (target: {7.0})")
    print(f"Average error after switch: {np.mean(np.abs(y[switch_step + 1:] - 7.0)):.4f}")


# ======================
# MAIN EXECUTION
# ======================
def main():
    print("=" * 50)
    print("NEURAL NETWORK CONTROLLER FOR CONSTANT OUTPUT MAINTENANCE")
    print("=" * 50)

    # Check if model already exists
    model_path = os.path.join(OUTPUT_DIR, 'controller_model.pth')

    if os.path.exists(model_path):
        print(f"\n[1/2] Found existing model at {model_path}")
        print("[2/2] Testing controller performance with existing model...")
        test_model(
            model_path=model_path,
            seq_length=TEST_SEQ_LENGTH,
            switch_step=SWITCH_STEP
        )
        print("\nPROCESS COMPLETED SUCCESSFULLY!")
        print(f"Results are available in the '{OUTPUT_DIR}' directory")
        print(f"Using pre-trained model: {model_path}")
    else:
        # Generate training data
        print("\n[1/4] Generating training data...")
        inputs, targets = generate_data(
            num_sequences=NUM_SEQUENCES,
            seq_length=SEQ_LENGTH,
            c_min=C_MIN,
            c_max=C_MAX
        )
        print(f"Generated {len(inputs)} training samples")

        # Create and train model
        print("\n[2/4] Creating and training model...")
        model = Controller(input_size=5, hidden_size=HIDDEN_SIZE)
        model = train_model(
            model,
            inputs,
            targets,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            batch_size=BATCH_SIZE
        )

        # Save trained model
        torch.save(model.state_dict(), model_path)
        print(f"\n[3/4] Model saved to {model_path}")

        # Test the controller
        print("\n[4/4] Testing controller performance...")
        test_model(
            model_path=model_path,
            seq_length=TEST_SEQ_LENGTH,
            switch_step=SWITCH_STEP
        )

        print("\nPROCESS COMPLETED SUCCESSFULLY!")
        print(f"Results are available in the '{OUTPUT_DIR}' directory")
        print(f"New model trained and saved to: {model_path}")


if __name__ == "__main__":
    main()
