# antra-uzduotis

Šis projektas demonstruoja, kaip apmokyti neuroninio tinklo (NT) valdiklį, kad jis palaikytų pastovią sistemos išvesties reikšmę y, parenkant tinkamą įvesties reikšmę x.
Sistema yra aprašoma šiomis matematinėmis lygtimis:
y(i) = y(i-1) + 0.01 * y(i-2) + 8 * x(i-1) - 0.3 * x(i-2) + 0.1 * z(i-1)
z(i) = z(i-1) + 2 * x(i-1) + 0.11
kur i yra laiko momentas.
Paleidimo instrukcija
1. Reikalavimai
Įsitikinkite, kad turite įdiegtą Python ir šias bibliotekas:
torch
numpy
matplotlib
Jei kuri nors biblioteka nėra įdiegta, galite tai padaryti naudodami pip:
code
Bash
pip install torch numpy matplotlib
2. Skripto vykdymas
Tiesiog paleiskite Python skriptą iš savo terminalo:
code
Bash
python tavo_failo_pavadinimas.py
Pakeiskite tavo_failo_pavadinimas.py į realų failo pavadinimą.
3. Veikimo principas
Pirmas paleidimas: Jei apmokytas modelis (controller_model.pth) neegzistuoja controller_results aplanke, skriptas pradės nuo duomenų generavimo ir modelio apmokymo. Šis procesas gali užtrukti kelias minutes. Baigus apmokymą, modelis bus išsaugotas ir iškart po to įvykdytas testas.
Sekantys paleidimai: Jei apmokytas modelis jau egzistuoja, skriptas praleis apmokymo žingsnį ir iš karto naudos esamą modelį testavimui.
4. Rezultatai
Visi rezultatai, įskaitant apmokymo nuostolių grafiką (training_loss.png), kontroliuojamos išvesties grafiką (controlled_output.png) ir apmokytą modelį (controller_model.pth), bus išsaugoti controller_results aplanke.
Konfigūruojami parametrai
Skripto pradžioje yra CONFIGURABLE PARAMETERS sekcija, kurioje galite keisti įvairius parametrus, norėdami paveikti modelio apmokymą ir testavimą.
Duomenų generavimo parametrai
NUM_SEQUENCES: Simuliacijos sekų skaičius, naudojamas apmokymo duomenims generuoti. Didesnis skaičius gali pagerinti modelio tikslumą, bet pailgins duomenų generavimo laiką.
SEQ_LENGTH: Kiekvienos simuliacijos sekos ilgis.
C_MIN, C_MAX: Tikslinės y reikšmės ribos (minimali ir maksimali) apmokymo metu.
Modelio parametrai
HIDDEN_SIZE: Paslėptųjų sluoksnių dydis neuroniniame tinkle. Didesnė reikšmė gali leisti modeliui išmokti sudėtingesnes priklausomybes, bet padidina skaičiavimo resursų poreikį.
Apmokymo parametrai
EPOCHS: Apmokymo epochų skaičius. Didesnis skaičius paprastai pagerina modelio tikslumą, bet reikalauja daugiau laiko.
LEARNING_RATE: Mokymosi greitis optimizatoriui. Šis parametras kontroliuoja, kaip greitai modelis adaptuojasi prie duomenų.
BATCH_SIZE: Duomenų paketo dydis apmokymo metu.
Testavimo parametrai
TEST_SEQ_LENGTH: Testavimo sekos ilgis.
SWITCH_STEP: Laiko žingsnis, kuriuo pakeičiama tikslinė y reikšmė (nuo 5.0 iki 7.0).