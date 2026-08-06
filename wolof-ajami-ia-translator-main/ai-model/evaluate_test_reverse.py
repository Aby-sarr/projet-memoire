import json
import random
import numpy as np
import torch

from config.config import (
    WOLOF_VOCAB_FILE,
    AJAMI_VOCAB_FILE,
    MODEL_DIR,
    EMBEDDING,
    HIDDEN_SIZE,
    DEVICE,
)

from models.encoder import EncoderGRU
from models.decoder import DecoderGRU
from models.attention import BahdanauAttention
from models.seq2seq import Seq2Seq


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
MAX_LENGTH = 100

MODEL_PATH = MODEL_DIR / "best_model_reverse.pt"


# ============================================================
# REPRODUCTIBILITE
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# AFFICHAGE
# ============================================================

print("=" * 60)
print("EVALUATION FINALE - AJAMI -> WOLOF LATIN")
print("=" * 60)

print("Device :", DEVICE)
print("Seed   :", SEED)
print("Model  :", MODEL_PATH)


# ============================================================
# CHARGEMENT VOCABULAIRE
# ============================================================

print("\n" + "=" * 60)
print("CHARGEMENT DES VOCABULAIRES")
print("=" * 60)


def load_vocab(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    stoi = data["stoi"]
    itos = data["itos"]

    # Certains fichiers peuvent stocker itos
    # sous forme de dictionnaire.
    if isinstance(itos, dict):

        itos = {
            int(k): v
            for k, v in itos.items()
        }

    return stoi, itos


ajami_stoi, ajami_itos = load_vocab(
    AJAMI_VOCAB_FILE
)

wolof_stoi, wolof_itos = load_vocab(
    WOLOF_VOCAB_FILE
)


print(
    "Vocabulaire source Ajami :",
    len(ajami_stoi)
)

print(
    "Vocabulaire cible Wolof :",
    len(wolof_stoi)
)


# ============================================================
# VERIFICATION VOCABULAIRE
# ============================================================

required_tokens = [
    "<PAD>",
    "<SOS>",
    "<EOS>",
    "<UNK>",
]


print("\n" + "=" * 60)
print("VERIFICATION DES TOKENS SPECIAUX")
print("=" * 60)


for token in required_tokens:

    if token not in ajami_stoi:

        raise ValueError(
            f"Token Ajami manquant : {token}"
        )

    if token not in wolof_stoi:

        raise ValueError(
            f"Token Wolof manquant : {token}"
        )


print("Tokens spéciaux Ajami : OK")
print("Tokens spéciaux Wolof : OK")


AJAMI_PAD_IDX = ajami_stoi["<PAD>"]
AJAMI_SOS_IDX = ajami_stoi["<SOS>"]
AJAMI_EOS_IDX = ajami_stoi["<EOS>"]
AJAMI_UNK_IDX = ajami_stoi["<UNK>"]

WOLOF_PAD_IDX = wolof_stoi["<PAD>"]
WOLOF_SOS_IDX = wolof_stoi["<SOS>"]
WOLOF_EOS_IDX = wolof_stoi["<EOS>"]
WOLOF_UNK_IDX = wolof_stoi["<UNK>"]


print("\nIndices des tokens :")

print(
    "Ajami <PAD> =",
    AJAMI_PAD_IDX
)

print(
    "Ajami <SOS> =",
    AJAMI_SOS_IDX
)

print(
    "Ajami <EOS> =",
    AJAMI_EOS_IDX
)

print(
    "Wolof <PAD> =",
    WOLOF_PAD_IDX
)

print(
    "Wolof <SOS> =",
    WOLOF_SOS_IDX
)

print(
    "Wolof <EOS> =",
    WOLOF_EOS_IDX
)


# ============================================================
# CONSTRUCTION MODELE
# ============================================================

print("\n" + "=" * 60)
print("CONSTRUCTION DU MODELE AJAMI -> WOLOF")
print("=" * 60)


INPUT_DIM = len(ajami_stoi)
OUTPUT_DIM = len(wolof_stoi)


encoder = EncoderGRU(
    input_dim=INPUT_DIM,
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE,
    pad_idx=AJAMI_PAD_IDX
)


attention = BahdanauAttention(
    HIDDEN_SIZE
)


decoder = DecoderGRU(
    output_dim=OUTPUT_DIM,
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)


model = Seq2Seq(
    encoder=encoder,
    decoder=decoder,
    attention=attention,
    device=DEVICE,
    src_pad_idx=AJAMI_PAD_IDX
).to(DEVICE)


print(
    "Modèle reverse créé avec succès !"
)


# ============================================================
# CHARGEMENT DU CHECKPOINT
# ============================================================

print("\n" + "=" * 60)
print("CHARGEMENT DU MEILLEUR MODELE REVERSE")
print("=" * 60)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Modèle introuvable : {MODEL_PATH}"
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


if isinstance(checkpoint, dict):

    print(
        "Epoch :",
        checkpoint.get("epoch", "?")
    )

    # Le checkpoint réel contient "valid_loss"
    print(
        "Loss validation :",
        checkpoint.get("valid_loss", "?")
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

else:

    state_dict = checkpoint

    print(
        "Checkpoint sous forme state_dict directe."
    )


# ============================================================
# COMPATIBILITE ANCIEN NOM encoder.rnn
# ============================================================

print("\n" + "-" * 60)
print("VERIFICATION DE COMPATIBILITE DU CHECKPOINT")
print("-" * 60)


converted_state_dict = {}

converted_count = 0


for key, value in state_dict.items():

    new_key = key

    if key.startswith("encoder.rnn."):

        new_key = key.replace(
            "encoder.rnn.",
            "encoder.gru.",
            1
        )

        converted_count += 1

    converted_state_dict[new_key] = value


print(
    "Paramètres encoder.rnn -> encoder.gru convertis :",
    converted_count
)


# ============================================================
# VERIFICATION DES CLES
# ============================================================

model_keys = set(
    model.state_dict().keys()
)

checkpoint_keys = set(
    converted_state_dict.keys()
)


missing_keys = model_keys - checkpoint_keys
unexpected_keys = checkpoint_keys - model_keys


if missing_keys:

    print("\nERREUR : clés manquantes :")

    for key in sorted(missing_keys):

        print(
            "  -",
            key
        )


if unexpected_keys:

    print("\nERREUR : clés inattendues :")

    for key in sorted(unexpected_keys):

        print(
            "  -",
            key
        )


if missing_keys or unexpected_keys:

    raise RuntimeError(
        "Le checkpoint n'est pas compatible "
        "avec l'architecture actuelle."
    )


model.load_state_dict(
    converted_state_dict,
    strict=True
)


model.eval()


print(
    "\nModèle reverse chargé avec succès !"
)


# ============================================================
# ENCODAGE AJAMI
# ============================================================

def encode_ajami(text):

    tokens = []

    for caractere in text:

        tokens.append(
            ajami_stoi.get(
                caractere,
                AJAMI_UNK_IDX
            )
        )

    # IMPORTANT :
    # Même convention que l'API reverse :
    #
    # Ajami + EOS
    #
    # et non :
    #
    # SOS + Ajami + EOS

    source_indices = (
        tokens
        + [AJAMI_EOS_IDX]
    )

    source = torch.tensor(
        source_indices,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    return source


# ============================================================
# DECODAGE WOLOF
# ============================================================

def decode_wolof(indices):

    chars = []

    for index in indices:

        index = int(index)

        if isinstance(wolof_itos, dict):

            caractere = wolof_itos.get(
                index,
                "<UNK>"
            )

        else:

            if 0 <= index < len(wolof_itos):

                caractere = wolof_itos[index]

            else:

                caractere = "<UNK>"

        if caractere == "<EOS>":

            break

        if caractere in [
            "<PAD>",
            "<SOS>",
            "<UNK>",
        ]:

            continue

        chars.append(
            caractere
        )

    return "".join(chars)


# ============================================================
# PREDICTION
# ============================================================

def predict(text, max_length=MAX_LENGTH):

    source = encode_ajami(text)


    with torch.no_grad():

        encoder_outputs, hidden = (
            model.encoder(source)
        )


    # Masque source
    src_mask = (
        source != model.src_pad_idx
    )


    input_token = torch.tensor(
        [WOLOF_SOS_IDX],
        dtype=torch.long,
        device=DEVICE
    )


    predicted_indices = []


    with torch.no_grad():

        for step in range(max_length):

            context, attention_weights = (
                model.attention(
                    hidden,
                    encoder_outputs,
                    mask=src_mask
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
                or predicted_token >= len(wolof_itos)
            ):

                break


            predicted_indices.append(
                predicted_token
            )


            if predicted_token == WOLOF_EOS_IDX:

                break


            input_token = torch.tensor(
                [predicted_token],
                dtype=torch.long,
                device=DEVICE
            )


    prediction = decode_wolof(
        predicted_indices
    )


    return prediction, predicted_indices


# ============================================================
# DISTANCE DE LEVENSHTEIN
# ============================================================

def levenshtein_distance(
    reference,
    prediction
):

    m = len(reference)
    n = len(prediction)

    previous = list(
        range(n + 1)
    )


    for i in range(1, m + 1):

        current = [
            i
        ] + [0] * n


        for j in range(1, n + 1):

            insertion = (
                current[j - 1] + 1
            )

            deletion = (
                previous[j] + 1
            )

            substitution = (
                previous[j - 1]
                + (
                    reference[i - 1]
                    != prediction[j - 1]
                )
            )

            current[j] = min(
                insertion,
                deletion,
                substitution
            )


        previous = current


    return previous[n]


# ============================================================
# CER
# ============================================================

def character_error_rate(
    reference,
    prediction
):

    if len(reference) == 0:

        if len(prediction) == 0:

            return 0.0

        return 1.0


    distance = levenshtein_distance(
        reference,
        prediction
    )


    return distance / len(reference)


# ============================================================
# TESTS CIBLES
# ============================================================

tests = [

    {
        "ajami": "ندانك",
        "reference": "ndank",
    },

    {
        "ajami": "خام",
        "reference": "xam",
    },

    {
        "ajami": "نيت",
        "reference": "nit",
    },

    {
        "ajami": "جاڠ",
        "reference": "jàng",
    },

]


print("\n" + "=" * 60)
print("TESTS CIBLES")
print("=" * 60)


results = []


for i, test in enumerate(
    tests,
    start=1
):

    ajami = test["ajami"]
    reference = test["reference"]


    print("\n" + "-" * 60)
    print(
        f"TEST {i}/{len(tests)}"
    )


    prediction, predicted_indices = (
        predict(ajami)
    )


    distance = levenshtein_distance(
        reference,
        prediction
    )


    cer = character_error_rate(
        reference,
        prediction
    )


    exact_match = (
        reference == prediction
    )


    print(
        "Ajami      :",
        ajami
    )


    print(
        "Unicode    :",
        ajami.encode(
            "unicode_escape"
        ).decode()
    )


    print(
        "Référence  :",
        reference
    )


    print(
        "Prediction :",
        prediction
    )


    print(
        "Unicode pred.:",
        prediction.encode(
            "unicode_escape"
        ).decode()
    )


    print(
        "Indices     :",
        predicted_indices
    )


    print(
        "Distance    :",
        distance
    )


    print(
        "CER         :",
        f"{cer:.4f}"
    )


    print(
        "Exact match :",
        exact_match
    )


    results.append(
        {
            "ajami": ajami,
            "reference": reference,
            "prediction": prediction,
            "distance": distance,
            "cer": cer,
            "exact_match": exact_match,
        }
    )


# ============================================================
# RESUME
# ============================================================

number_tests = len(results)


exact_matches = sum(
    1
    for result in results
    if result["exact_match"]
)


exact_match_percentage = (
    100.0 * exact_matches / number_tests
    if number_tests > 0
    else 0.0
)


mean_cer = (
    sum(
        result["cer"]
        for result in results
    )
    / number_tests
    if number_tests > 0
    else 0.0
)


print("\n" + "=" * 60)
print("RESUME DES TESTS CIBLES")
print("=" * 60)


print(
    "Nombre de tests :",
    number_tests
)


print(
    "Exact match     :",
    f"{exact_matches} / {number_tests}"
)


print(
    "Exact match (%) :",
    f"{exact_match_percentage:.2f}%"
)


print(
    "CER moyen       :",
    f"{mean_cer:.4f}"
)


print("\n" + "=" * 60)
print("EVALUATION TERMINEE")
print("=" * 60)