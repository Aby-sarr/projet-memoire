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
    title="Wolof Latin <-> Wolof Ajami API",
    version="2.0.0"
)


# ============================================================
# CHARGEMENT DES VOCABULAIRES
# ============================================================

print("=" * 70)
print("DEMARRAGE DE L'API WOLOF LATIN <-> WOLOF AJAMI")
print("=" * 70)

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


# ============================================================
# NORMALISATION DES itos
# ============================================================

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


print(
    "Vocabulaire Wolof :",
    len(wolof_stoi)
)

print(
    "Vocabulaire Ajami :",
    len(ajami_stoi)
)


# ============================================================
# VERIFICATION DES TOKENS
# ============================================================

required_wolof_tokens = [
    "<PAD>",
    "<SOS>",
    "<EOS>",
    "<UNK>",
]

required_ajami_tokens = [
    "<PAD>",
    "<SOS>",
    "<EOS>",
    "<UNK>",
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


print("Tokens speciaux : OK")


# ============================================================
# MODELE LATIN -> AJAMI
# ============================================================

print()
print("=" * 70)
print("CONSTRUCTION DU MODELE LATIN -> AJAMI")
print("=" * 70)


encoder_lat2ajami = EncoderGRU(
    input_dim=len(wolof_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE,
    pad_idx=wolof_stoi["<PAD>"]
)


attention_lat2ajami = BahdanauAttention(
    HIDDEN_SIZE
)


decoder_lat2ajami = DecoderGRU(
    output_dim=len(ajami_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)


model_lat2ajami = Seq2Seq(
    encoder_lat2ajami,
    decoder_lat2ajami,
    attention_lat2ajami,
    DEVICE,
    src_pad_idx=wolof_stoi["<PAD>"]
).to(DEVICE)


MODEL_PATH_LAT2AJAMI = (
    MODEL_DIR / "best_model.pt"
)

if not MODEL_PATH_LAT2AJAMI.exists():

    raise FileNotFoundError(
        f"Modele Latin -> Ajami introuvable : "
        f"{MODEL_PATH_LAT2AJAMI}"
    )
checkpoint_lat2ajami = torch.load(
    MODEL_PATH_LAT2AJAMI,
    map_location=DEVICE
)
if (
    isinstance(checkpoint_lat2ajami, dict)
    and "model_state_dict" in checkpoint_lat2ajami
):

    model_lat2ajami.load_state_dict(
        checkpoint_lat2ajami["model_state_dict"]
    )

else:

    model_lat2ajami.load_state_dict(
        checkpoint_lat2ajami
    )


model_lat2ajami.eval()


print(
    "best_model.pt charge avec succes !"
)


# ============================================================
# MODELE AJAMI -> LATIN
# ============================================================

print()
print("=" * 70)
print("CONSTRUCTION DU MODELE AJAMI -> LATIN")
print("=" * 70)


encoder_ajami2lat = EncoderGRU(
    input_dim=len(ajami_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE,
    pad_idx=ajami_stoi["<PAD>"]
)


attention_ajami2lat = BahdanauAttention(
    HIDDEN_SIZE
)


decoder_ajami2lat = DecoderGRU(
    output_dim=len(wolof_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)


model_ajami2lat = Seq2Seq(
    encoder_ajami2lat,
    decoder_ajami2lat,
    attention_ajami2lat,
    DEVICE,
    src_pad_idx=ajami_stoi["<PAD>"]
).to(DEVICE)


MODEL_PATH_AJAMI2LAT = (
    MODEL_DIR / "best_model_reverse.pt"
)


if not MODEL_PATH_AJAMI2LAT.exists():

    raise FileNotFoundError(
        f"Modele Ajami -> Latin introuvable : "
        f"{MODEL_PATH_AJAMI2LAT}"
    )


checkpoint_ajami2lat = torch.load(
    MODEL_PATH_AJAMI2LAT,
    map_location=DEVICE
)


if (
    isinstance(checkpoint_ajami2lat, dict)
    and "model_state_dict" in checkpoint_ajami2lat
):

    state_dict_ajami2lat = (
        checkpoint_ajami2lat["model_state_dict"]
    )

else:

    state_dict_ajami2lat = checkpoint_ajami2lat


# ============================================================
# COMPATIBILITE ANCIEN MODELE REVERSE
# ============================================================

state_dict_ajami2lat = {
    key.replace(
        "encoder.rnn.",
        "encoder.gru."
    ): value

    for key, value in state_dict_ajami2lat.items()
}


model_ajami2lat.load_state_dict(
    state_dict_ajami2lat
)


model_ajami2lat.eval()


print(
    "best_model_reverse.pt charge avec succes !"
)

model_ajami2lat.eval()


print(
    "best_model_reverse.pt charge avec succes !"
)


print()
print("=" * 70)
print("LES DEUX MODELES SONT CHARGES")
print("=" * 70)


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
# LATIN -> AJAMI
# ============================================================

def predict_lat2ajami(
    sentence: str,
    max_length: int = MAX_LENGTH
) -> str:

    sentence = str(sentence)

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


    with torch.no_grad():

        encoder_outputs, hidden = (
            model_lat2ajami.encoder(source)
        )


    src_mask = (
        source != model_lat2ajami.encoder.pad_idx
    )


    input_token = torch.tensor(
        [ajami_stoi["<SOS>"]],
        dtype=torch.long,
        device=DEVICE
    )


    resultat = []


    with torch.no_grad():

        for step in range(max_length):

            context, attention_weights = (
                model_lat2ajami.attention(
                    hidden,
                    encoder_outputs,
                    src_mask
                )
            )


            output, hidden = (
                model_lat2ajami.decoder(
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


            predicted_char = (
                ajami_itos[predicted_token]
            )


            if predicted_char == "<EOS>":

                break


            if predicted_char not in [
                "<PAD>",
                "<SOS>",
                "<UNK>",
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
# AJAMI -> LATIN
# ============================================================

def predict_ajami2lat(
    sentence: str,
    max_length: int = MAX_LENGTH
) -> str:

    sentence = str(sentence)

    tokens = []


    # --------------------------------------------------------
    # Encodage Ajami
    # --------------------------------------------------------

    for caractere in sentence:

        if caractere in ajami_stoi:

            tokens.append(
                ajami_stoi[caractere]
            )

        else:

            tokens.append(
                ajami_stoi["<UNK>"]
            )


    # IMPORTANT :
    # Le dataset reverse utilise :
    #
    # Ajami + <EOS>
    #
    # et PAS :
    #
    # <SOS> + Ajami + <EOS>

    source_indices = (
        tokens
        + [ajami_stoi["<EOS>"]]
    )


    source = torch.tensor(
        source_indices,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)


    # --------------------------------------------------------
    # ENCODEUR
    # --------------------------------------------------------

    with torch.no_grad():

        encoder_outputs, hidden = (
            model_ajami2lat.encoder(source)
        )


    # --------------------------------------------------------
    # MASQUE SOURCE
    # --------------------------------------------------------

    src_mask = (
        source != model_ajami2lat.encoder.pad_idx
    )


    # --------------------------------------------------------
    # SOS COTE CIBLE
    # --------------------------------------------------------

    input_token = torch.tensor(
        [wolof_stoi["<SOS>"]],
        dtype=torch.long,
        device=DEVICE
    )


    resultat = []


    # --------------------------------------------------------
    # DECODAGE AUTO-REGRESSIF
    # --------------------------------------------------------

    with torch.no_grad():

        for step in range(max_length):

            context, attention_weights = (
                model_ajami2lat.attention(
                    hidden,
                    encoder_outputs,
                    src_mask
                )
            )


            output, hidden = (
                model_ajami2lat.decoder(
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
                or predicted_token >= len(wolof_itos)
            ):

                break


            predicted_char = (
                wolof_itos[predicted_token]
            )


            if predicted_char == "<EOS>":

                break


            if predicted_char not in [
                "<PAD>",
                "<SOS>",
                "<UNK>",
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
        "models": {
            "lat2ajami": "best_model.pt",
            "ajami2lat": "best_model_reverse.pt"
        },
        "device": str(DEVICE),
        "directions": [
            "lat2ajami",
            "ajami2lat"
        ]
    }


# ============================================================
# ROUTE LATIN -> AJAMI
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


    print()
    print(
        f"[LAT2AJAMI] Prediction recue : '{text}'"
    )


    result = predict_lat2ajami(text)


    print(
        f"[LAT2AJAMI] Resultat : '{result}'"
    )


    return PredictionResponse(
        input=text,
        output=result,
        direction="lat2ajami"
    )


# ============================================================
# ROUTE AJAMI -> LATIN
# ============================================================

@app.post(
    "/predict_reverse",
    response_model=PredictionResponse
)
def prediction_reverse(
    request: PredictionRequest
):

    text = request.text.strip()


    if not text:

        return PredictionResponse(
            input="",
            output="",
            direction="ajami2lat"
        )


    print()
    print(
        f"[AJAMI2LAT] Prediction recue : '{text}'"
    )


    result = predict_ajami2lat(text)


    print(
        f"[AJAMI2LAT] Resultat : '{result}'"
    )


    return PredictionResponse(
        input=text,
        output=result,
        direction="ajami2lat"
    )