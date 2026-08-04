from pathlib import Path
import json

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from config.config import (
    WOLOF_VOCAB_FILE,
    AJAMI_VOCAB_FILE,
    MODEL_DIR,
    EMBEDDING,
    HIDDEN_SIZE,
    DEVICE,
)

from models.encoder import EncoderGRU
from models.attention import BahdanauAttention
from models.decoder import DecoderGRU
from models.seq2seq import Seq2Seq


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
MAX_LENGTH = 100

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# APPLICATION FASTAPI
# ============================================================

app = FastAPI(
    title="Wolof Latin → Wolof Ajami API",
    version="1.0.0"
)


# ============================================================
# CHARGEMENT DES VOCABULAIRES
# ============================================================

print("=" * 60)
print("DEMARRAGE DE L'API WOLOF LATIN -> WOLOF AJAMI")
print("=" * 60)

print("Device :", DEVICE)

with open(
    WOLOF_VOCAB_FILE,
    "r",
    encoding="utf-8"
) as f:
    wolof_vocab = json.load(f)


with open(
    AJAMI_VOCAB_FILE,
    "r",
    encoding="utf-8"
) as f:
    ajami_vocab = json.load(f)


wolof_stoi = wolof_vocab["stoi"]
wolof_itos = wolof_vocab["itos"]

ajami_stoi = ajami_vocab["stoi"]
ajami_itos = ajami_vocab["itos"]


# Certains fichiers JSON peuvent stocker itos comme dictionnaire.
if isinstance(wolof_itos, dict):
    wolof_itos = {
        int(index): caractere
        for index, caractere in wolof_itos.items()
    }


if isinstance(ajami_itos, dict):
    ajami_itos = {
        int(index): caractere
        for index, caractere in ajami_itos.items()
    }


print("Vocabulaire Wolof :", len(wolof_stoi))
print("Vocabulaire Ajami :", len(ajami_stoi))


# ============================================================
# VERIFICATION DES TOKENS
# ============================================================

required_wolof_tokens = [
    "<PAD>",
    "<SOS>",
    "<EOS>",
    "<UNK>"
]

required_ajami_tokens = [
    "<PAD>",
    "<SOS>",
    "<EOS>"
]


for token in required_wolof_tokens:
    if token not in wolof_stoi:
        raise ValueError(
            f"Token Wolof manquant : {token}"
        )


for token in required_ajami_tokens:
    if token not in ajami_stoi:
        raise ValueError(
            f"Token Ajami manquant : {token}"
        )


print("Tokens spéciaux : OK")


# ============================================================
# CONSTRUCTION DU MODELE
# ============================================================

print()
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
    output_dim=len(ajami_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)


model = Seq2Seq(
    encoder,
    decoder,
    attention,
    DEVICE
).to(DEVICE)


# ============================================================
# CHARGEMENT DU MEILLEUR MODELE
# ============================================================

MODEL_PATH = MODEL_DIR / "best_model.pt"


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Modèle introuvable : {MODEL_PATH}"
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print("Modèle chargé avec succès !")


if "epoch" in checkpoint:
    print(
        "Meilleure epoch :",
        checkpoint["epoch"]
    )


if "valid_loss" in checkpoint:
    print(
        "Loss validation :",
        checkpoint["valid_loss"]
    )

elif "val_loss" in checkpoint:
    print(
        "Loss validation :",
        checkpoint["val_loss"]
    )


print("=" * 60)


# ============================================================
# MODELE DE REQUETE
# ============================================================

class PredictionRequest(BaseModel):

    text: str


# ============================================================
# MODELE DE REPONSE
# ============================================================

class PredictionResponse(BaseModel):

    input: str
    output: str
    direction: str


# ============================================================
# FONCTION DE PREDICTION
# ============================================================

def predict(
    sentence: str,
    max_length: int = MAX_LENGTH
) -> str:

    sentence = str(sentence)


    # --------------------------------------------------------
    # Encodage de la phrase Wolof Latin
    # --------------------------------------------------------

    tokens = []


    for caractere in sentence:

        if caractere in wolof_stoi:

            tokens.append(
                wolof_stoi[caractere]
            )

        else:

            tokens.append(
                wolof_stoi["<UNK>"]
            )


    source_indices = (
        [wolof_stoi["<SOS>"]]
        + tokens
        + [wolof_stoi["<EOS>"]]
    )


    source = torch.tensor(
        source_indices,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)


    # --------------------------------------------------------
    # ENCODER
    # --------------------------------------------------------

    with torch.no_grad():

        encoder_outputs, hidden = (
            model.encoder(source)
        )


    # --------------------------------------------------------
    # Premier token du decoder
    # --------------------------------------------------------

    input_token = torch.tensor(
        [ajami_stoi["<SOS>"]],
        dtype=torch.long,
        device=DEVICE
    )


    resultat = []


    # --------------------------------------------------------
    # DECODAGE
    # --------------------------------------------------------

    for step in range(max_length):

        with torch.no_grad():

            context, attention_weights = (
                model.attention(
                    hidden,
                    encoder_outputs
                )
            )


            output, hidden = (
                model.decoder(
                    input_token,
                    hidden,
                    context
                )
            )


        predicted_token = (
            output.argmax(
                dim=1
            ).item()
        )


        if (
            predicted_token < 0
            or predicted_token >= len(ajami_itos)
        ):
            break


        predicted_char = ajami_itos[
            predicted_token
        ]


        if predicted_char == "<EOS>":
            break


        if predicted_char not in [
            "<PAD>",
            "<SOS>"
        ]:

            resultat.append(
                predicted_char
            )


        input_token = torch.tensor(
            [predicted_token],
            dtype=torch.long,
            device=DEVICE
        )


    return "".join(resultat)


# ============================================================
# ROUTE HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "UP",
        "model": "best_model.pt",
        "device": str(DEVICE),
        "direction": "lat2ajami"
    }


# ============================================================
# ROUTE PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def prediction(request: PredictionRequest):

    text = request.text.strip()


    if not text:

        return PredictionResponse(
            input="",
            output="",
            direction="lat2ajami"
        )


    print(
        f"Prediction reçue : '{text}'"
    )


    result = predict(text)


    print(
        f"Résultat : '{result}'"
    )


    return PredictionResponse(
        input=text,
        output=result,
        direction="lat2ajami"
    )