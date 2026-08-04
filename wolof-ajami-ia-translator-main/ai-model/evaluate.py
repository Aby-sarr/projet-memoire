import torch
import json
import pandas as pd

from config.config import (
    WOLOF_VOCAB_FILE,
    AJAMI_VOCAB_FILE,
    MODEL_DIR,
    CORPUS_FILE,
    EMBEDDING,
    HIDDEN_SIZE,
    DEVICE,
)

from models.encoder import EncoderGRU
from models.attention import BahdanauAttention
from models.decoder import DecoderGRU
from models.seq2seq import Seq2Seq


# ==================================================
# 1. CHARGEMENT DES VOCABULAIRES
# ==================================================

print("=" * 50)
print("Chargement des vocabulaires...")

with open(WOLOF_VOCAB_FILE, "r", encoding="utf-8") as f:
    wolof_vocab = json.load(f)

with open(AJAMI_VOCAB_FILE, "r", encoding="utf-8") as f:
    ajami_vocab = json.load(f)

wolof_stoi = wolof_vocab["stoi"]
ajami_itos = ajami_vocab["itos"]

print("Vocabulaire Wolof :", len(wolof_stoi))
print("Vocabulaire Ajami :", len(ajami_itos))


# ==================================================
# 2. CONSTRUCTION DU MODELE
# ==================================================

print("=" * 50)
print("Construction du modèle...")

encoder = EncoderGRU(
    input_dim=len(wolof_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)

attention = BahdanauAttention(
    HIDDEN_SIZE
)

decoder = DecoderGRU(
    output_dim=len(ajami_itos),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)

model = Seq2Seq(
    encoder,
    decoder,
    attention,
    DEVICE
).to(DEVICE)


# ==================================================
# 3. CHARGEMENT DU MODELE
# ==================================================

MODEL_PATH = MODEL_DIR / "best_model.pt"

print("=" * 50)
print("Chargement du modèle...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Modèle chargé avec succès !")


# ==================================================
# 4. CHARGEMENT DU CORPUS
# ==================================================

print("=" * 50)
print("Chargement du corpus...")

df = pd.read_csv(
    CORPUS_FILE
)

print("Nombre total de paires :", len(df))

# On prend seulement 100 phrases
df = df.head(100)

print("Nombre de phrases évaluées :", len(df))


# ==================================================
# 5. FONCTION DE PREDICTION
# ==================================================

def predict(sentence):

    tokens = []

    for caractere in sentence:

        if caractere in wolof_stoi:
            tokens.append(wolof_stoi[caractere])
        else:
            tokens.append(wolof_stoi["<UNK>"])

    source_indices = [
        wolof_stoi["<SOS>"]
    ] + tokens + [
        wolof_stoi["<EOS>"]
    ]

    source = torch.tensor(
        source_indices,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    MAX_LENGTH = 100

    target = torch.full(
        (1, MAX_LENGTH),
        ajami_itos.index("<PAD>"),
        dtype=torch.long,
        device=DEVICE
    )

    target[0, 0] = ajami_itos.index("<SOS>")

    with torch.no_grad():

        output = model(
            source,
            target,
            teacher_forcing_ratio=0
        )

    predictions = output.argmax(
        dim=2
    )

    resultat = []

    for indice in predictions[0]:

        caractere = ajami_itos[indice.item()]

        if caractere == "<EOS>":
            break

        if caractere in [
            "<PAD>",
            "<SOS>"
        ]:
            continue

        resultat.append(caractere)

    return "".join(resultat)


# ==================================================
# 6. EVALUATION
# ==================================================

print("=" * 50)
print("Début de l'évaluation...")

correct = 0
total = 0

for index, row in df.iterrows():

    wolof = str(row["Wolof"])
    ajami_attendu = str(row["ajami"])

    ajami_pred = predict(wolof)

    if ajami_pred == ajami_attendu:
        correct += 1

    total += 1

    if total <= 10:

        print()
        print("Exemple", total)
        print("Wolof    :", wolof)
        print("Attendu  :", ajami_attendu)
        print("Prédit   :", ajami_pred)

# ACCURACY
accuracy = (
    correct / total
) * 100

print("=" * 50)
print("RESULTATS")
print("=" * 50)

print(
    f"Accuracy : {accuracy:.2f}%"
)

print(
    f"Corrects : {correct}/{total}"
)

print("=" * 50)
print("Evaluation terminée.")