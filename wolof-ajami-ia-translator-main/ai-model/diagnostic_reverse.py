import json
import random
import torch

from config.config import (
    WOLOF_VOCAB_FILE,
    AJAMI_VOCAB_FILE,
    MODEL_DIR,
)

from models.encoder import EncoderGRU
from models.decoder import DecoderGRU
from models.attention import BahdanauAttention
from models.seq2seq import Seq2Seq


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBEDDING_DIM = 256
HIDDEN_DIM = 512

MODEL_PATH = MODEL_DIR / "best_model_reverse.pt"


# ============================================================
# CHARGEMENT VOCABULAIRE
# ============================================================

def load_vocab(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stoi = data["stoi"]
    itos = data["itos"]

    # Le vocabulaire Wolof possède itos sous forme de liste.
    # Certains vocabulaires peuvent avoir itos sous forme de dict.
    if isinstance(itos, dict):
        itos = {
            int(k): v
            for k, v in itos.items()
        }

    return stoi, itos


# ============================================================
# CHARGEMENT
# ============================================================

ajami_stoi, ajami_itos = load_vocab(AJAMI_VOCAB_FILE)
wolof_stoi, wolof_itos = load_vocab(WOLOF_VOCAB_FILE)


print("=" * 60)
print("DIAGNOSTIC MODELE REVERSE")
print("AJAMI -> WOLOF LATIN")
print("=" * 60)

print("Device :", DEVICE)

print("\n" + "=" * 60)
print("VOCABULAIRES")
print("=" * 60)

print("Vocabulaire Ajami :", len(ajami_stoi))
print("Vocabulaire Wolof :", len(wolof_stoi))


# ============================================================
# TOKENS
# ============================================================

PAD_IDX = wolof_stoi["<PAD>"]
SOS_IDX = wolof_stoi["<SOS>"]
EOS_IDX = wolof_stoi["<EOS>"]
UNK_IDX = wolof_stoi["<UNK>"]

print("\n" + "=" * 60)
print("TOKENS WOLOF")
print("=" * 60)

for token in [
    "<PAD>",
    "<SOS>",
    "<EOS>",
    "<UNK>",
    "C",
    "c",
    "g",
    "d",
    "ë",
    "é",
    "ŋ",
]:
    if token in wolof_stoi:
        print(
            repr(token),
            "=>",
            wolof_stoi[token]
        )


# ============================================================
# CONSTRUCTION MODELE
# ============================================================

print("\n" + "=" * 60)
print("CONSTRUCTION DU MODELE REVERSE")
print("=" * 60)

INPUT_DIM = len(ajami_stoi)
OUTPUT_DIM = len(wolof_stoi)

encoder = EncoderGRU(
    input_dim=INPUT_DIM,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
)

attention = BahdanauAttention(
    hidden_dim=HIDDEN_DIM
)

decoder = DecoderGRU(
    output_dim=OUTPUT_DIM,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
)

model = Seq2Seq(
    encoder=encoder,
    decoder=decoder,
    attention=attention,
    device=DEVICE
).to(DEVICE)


print("Modèle reverse créé avec succès !")


# ============================================================
# CHARGEMENT MODELE
# ============================================================

print("\n" + "=" * 60)
print("CHARGEMENT DE BEST_MODEL_REVERSE")
print("=" * 60)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Epoch :",
        checkpoint.get("epoch", "?")
    )

    print(
        "Loss validation :",
        checkpoint.get("val_loss", "?")
    )

else:

    model.load_state_dict(checkpoint)

    print("Checkpoint chargé directement.")


model.eval()

print("Modèle reverse chargé avec succès !")


# ============================================================
# ENCODAGE AJAMI
# ============================================================

def encode_ajami(text):

    ids = [
        ajami_stoi.get(
            char,
            ajami_stoi["<UNK>"]
        )
        for char in text
    ]

    ids = [
        ajami_stoi["<SOS>"]
    ] + ids + [
        ajami_stoi["<EOS>"]
    ]

    return torch.tensor(
        ids,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)


# ============================================================
# DECODAGE CORRIGE
# ============================================================

def decode_wolof(indices):

    chars = []

    for indice in indices:

        indice = int(indice)

        # IMPORTANT :
        # wolof_itos est une LISTE.
        # On ne doit PAS utiliser .get()
        if 0 <= indice < len(wolof_itos):
            caractere = wolof_itos[indice]
        else:
            caractere = "<UNK>"

        # Ignorer tokens spéciaux
        if caractere in [
            "<PAD>",
            "<SOS>"
        ]:
            continue

        if caractere == "<EOS>":
            break

        chars.append(caractere)

    return "".join(chars)


# ============================================================
# PREDICTION
# ============================================================

def predict(text, max_len=100):

    src = encode_ajami(text)

    with torch.no_grad():

        encoder_outputs, hidden = model.encoder(src)

        input_token = torch.tensor(
            [ajami_stoi["<SOS>"] if False else wolof_stoi["<SOS>"]],
            dtype=torch.long,
            device=DEVICE
        )

        predicted_indices = []

        for step in range(max_len):

            context, attention_weights = model.attention(
                hidden,
                encoder_outputs
            )

            output, hidden = model.decoder(
                input_token,
                hidden,
                context
            )

            prediction = output.argmax(1)

            predicted_idx = prediction.item()

            predicted_indices.append(
                predicted_idx
            )

            if predicted_idx == EOS_IDX:
                break

            input_token = prediction

    return predicted_indices


# ============================================================
# DIAGNOSTIC
# ============================================================

def diagnostic_prediction(sentence):

    print("\n" + "-" * 60)
    print("ENTREE AJAMI")
    print("-" * 60)

    print(sentence)

    predicted_indices = predict(
        sentence,
        max_len=100
    )

    print("\n" + "-" * 60)
    print("INDICES PREDITS")
    print("-" * 60)

    print(predicted_indices)

    print("\n" + "-" * 60)
    print("DETAIL DES PREDICTIONS")
    print("-" * 60)

    for position, indice in enumerate(
        predicted_indices
    ):

        indice = int(indice)

        # CORRECTION PRINCIPALE
        if 0 <= indice < len(wolof_itos):
            caractere = wolof_itos[indice]
        else:
            caractere = "<UNK>"

        print(
            f"Position {position:03d} | "
            f"Indice {indice:03d} | "
            f"Caractère {repr(caractere)}"
        )

    prediction = decode_wolof(
        predicted_indices
    )

    print("\n" + "-" * 60)
    print("PREDICTION FINALE")
    print("-" * 60)

    print(prediction)


# ============================================================
# TESTS
# ============================================================

tests = [
    "گوددي",
    "دێپپ",
    "يوخوي",
    "چێرام",
    "باريوول",
    "كو نەكك چي توول بي",
    "ماا ڠي گيس چي تێەرە بي",
    "نوونو مو فاب كااس",
]


print("\n" + "=" * 60)
print("TESTS CIBLES")
print("=" * 60)

for i, sentence in enumerate(
    tests,
    start=1
):

    print(
        f"\nTEST {i}/{len(tests)}"
    )

    diagnostic_prediction(
        sentence
    )


print("\n" + "=" * 60)
print("DIAGNOSTIC TERMINE")
print("=" * 60)