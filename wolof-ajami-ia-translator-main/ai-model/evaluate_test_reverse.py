import json
import random
import numpy as np
import pandas as pd
import torch

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
from models.decoder import DecoderGRU
from models.attention import BahdanauAttention
from models.seq2seq import Seq2Seq


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
MAX_LENGTH = 100

MODEL_PATH = MODEL_DIR / "best_model_reverse.pt"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 60)
print("EVALUATION FINALE - AJAMI -> WOLOF LATIN")
print("=" * 60)

print("Device :", DEVICE)
print("Seed   :", SEED)
print("Model  :", MODEL_PATH)


# ============================================================
# 2. CHARGEMENT DES VOCABULAIRES
# ============================================================

print()
print("=" * 60)
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

    if isinstance(itos, dict):

        itos = {
            int(index): caractere
            for index, caractere in itos.items()
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
# 3. VERIFICATION DES TOKENS SPECIAUX
# ============================================================

print()
print("=" * 60)
print("VERIFICATION DES TOKENS SPECIAUX")
print("=" * 60)


required_tokens = [
    "<PAD>",
    "<SOS>",
    "<EOS>",
    "<UNK>",
]


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


print()
print("Indices des tokens :")

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
# 4. CONSTRUCTION DU MODELE AJAMI -> WOLOF
# ============================================================

print()
print("=" * 60)
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
# 5. CHARGEMENT DU MEILLEUR MODELE REVERSE
# ============================================================

print()
print("=" * 60)
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

    print(
        "Loss validation :",
        checkpoint.get(
            "valid_loss",
            "?"
        )
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
# 6. COMPATIBILITE DU CHECKPOINT
# ============================================================

print()
print("-" * 60)
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
# 7. VERIFICATION DES CLES
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

    print()
    print("ERREUR : clés manquantes :")

    for key in sorted(missing_keys):

        print(
            "  -",
            key
        )


if unexpected_keys:

    print()
    print("ERREUR : clés inattendues :")

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
# 8. CHARGEMENT DU CORPUS
# ============================================================

print()
print("=" * 60)
print("CHARGEMENT DU CORPUS")
print("=" * 60)


df = pd.read_csv(
    CORPUS_FILE
)

df = df.reset_index(
    drop=True
)


print(
    "Nombre total de phrases :",
    len(df)
)


# ============================================================
# 9. SEPARATION TRAIN / VALIDATION / TEST
# ============================================================

print()
print("=" * 60)
print("SEPARATION TRAIN / VALIDATION / TEST")
print("=" * 60)


generator = torch.Generator().manual_seed(
    SEED
)


indices = torch.randperm(
    len(df),
    generator=generator
).tolist()


total = len(df)


train_size = int(
    0.80 * total
)


valid_size = int(
    0.10 * total
)


train_indices = indices[
    :train_size
]


valid_indices = indices[
    train_size:
    train_size + valid_size
]


test_indices = indices[
    train_size + valid_size:
]


print(
    "Train :",
    len(train_indices)
)

print(
    "Validation :",
    len(valid_indices)
)

print(
    "Test :",
    len(test_indices)
)


test_df = df.iloc[
    test_indices
].reset_index(
    drop=True
)


print()

print(
    "Nombre de phrases utilisées pour le test :",
    len(test_df)
)


# ============================================================
# 10. ENCODAGE AJAMI
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


    # Même convention que votre API reverse :
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
# 11. DECODAGE WOLOF
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
# 12. PREDICTION AJAMI -> WOLOF
# ============================================================

def predict(
    text,
    max_length=MAX_LENGTH
):

    source = encode_ajami(
        text
    )


    # --------------------------------------------------------
    # Encodeur
    # --------------------------------------------------------

    with torch.no_grad():

        encoder_outputs, hidden = (
            model.encoder(source)
        )


    # --------------------------------------------------------
    # Masque de la source
    # --------------------------------------------------------

    src_mask = (
        source != model.src_pad_idx
    )


    # --------------------------------------------------------
    # Premier token du décodeur
    # --------------------------------------------------------

    input_token = torch.tensor(
        [WOLOF_SOS_IDX],
        dtype=torch.long,
        device=DEVICE
    )


    predicted_indices = []


    # --------------------------------------------------------
    # Décodage auto-régressif
    # --------------------------------------------------------

    with torch.no_grad():

        for step in range(
            max_length
        ):


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
                or predicted_token >= len(
                    wolof_itos
                )
            ):

                break


            predicted_indices.append(
                predicted_token
            )


            if (
                predicted_token
                == WOLOF_EOS_IDX
            ):

                break


            input_token = torch.tensor(
                [predicted_token],
                dtype=torch.long,
                device=DEVICE
            )


    prediction = decode_wolof(
        predicted_indices
    )


    return (
        prediction,
        predicted_indices
    )


# ============================================================
# 13. DISTANCE DE LEVENSHTEIN
# ============================================================

def levenshtein_distance(
    reference,
    hypothesis
):

    rows = len(reference) + 1
    cols = len(hypothesis) + 1


    distance = [
        [0] * cols
        for _ in range(rows)
    ]


    for i in range(rows):

        distance[i][0] = i


    for j in range(cols):

        distance[0][j] = j


    for i in range(
        1,
        rows
    ):

        for j in range(
            1,
            cols
        ):


            if (
                reference[i - 1]
                == hypothesis[j - 1]
            ):

                cost = 0

            else:

                cost = 1


            distance[i][j] = min(

                distance[i - 1][j] + 1,

                distance[i][j - 1] + 1,

                distance[i - 1][j - 1] + cost

            )


    return distance[-1][-1]


# ============================================================
# 14. EVALUATION DU JEU DE TEST
# ============================================================

print()
print("=" * 60)
print("DEBUT DE L'EVALUATION DU JEU DE TEST")
print("=" * 60)


correct = 0
total = 0


total_char_errors = 0
total_reference_chars = 0


total_word_errors = 0
total_reference_words = 0


errors = []


for index, row in test_df.iterrows():


    # --------------------------------------------------------
    # SOURCE = AJAMI
    # CIBLE = WOLOF
    # --------------------------------------------------------

    ajami = str(
        row["ajami"]
    )


    expected = str(
        row["Wolof"]
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction, predicted_indices = (
        predict(
            ajami
        )
    )


    # ========================================================
    # ACCURACY
    # ========================================================

    if prediction == expected:

        correct += 1


    # ========================================================
    # CER
    # ========================================================

    char_distance = levenshtein_distance(
        expected,
        prediction
    )


    total_char_errors += (
        char_distance
    )


    total_reference_chars += len(
        expected
    )


    # ========================================================
    # WER
    # ========================================================

    reference_words = (
        expected.split()
    )


    prediction_words = (
        prediction.split()
    )


    word_distance = levenshtein_distance(
        reference_words,
        prediction_words
    )


    total_word_errors += (
        word_distance
    )


    total_reference_words += len(
        reference_words
    )


    # ========================================================
    # ENREGISTREMENT DES ERREURS
    # ========================================================

    if prediction != expected:

        errors.append({

            "Ajami": ajami,

            "Wolof_attendu": expected,

            "Wolof_predit": prediction

        })


    total += 1


    # ========================================================
    # PROGRESSION
    # ========================================================

    if total % 100 == 0:

        print(
            f"Évaluation : "
            f"{total}/{len(test_df)} phrases..."
        )


# ============================================================
# 15. CALCUL DES METRIQUES
# ============================================================

accuracy = (
    correct / total
) * 100


if total_reference_chars > 0:

    cer = (
        total_char_errors
        / total_reference_chars
    ) * 100

else:

    cer = 0.0


if total_reference_words > 0:

    wer = (
        total_word_errors
        / total_reference_words
    ) * 100

else:

    wer = 0.0


incorrect = (
    total - correct
)


# ============================================================
# 16. RESULTATS FINAUX
# ============================================================

print()
print("=" * 60)
print("RESULTATS FINAUX - JEU DE TEST")
print("=" * 60)


print(
    "Nombre total de phrases :",
    total
)


print(
    "Phrases correctes :",
    correct
)


print(
    "Phrases incorrectes :",
    incorrect
)


print(
    f"Accuracy : {accuracy:.2f}%"
)


print(
    f"CER : {cer:.2f}%"
)


print(
    f"WER : {wer:.2f}%"
)


# ============================================================
# 17. EXEMPLES D'ERREURS
# ============================================================

print()
print("=" * 60)
print("EXEMPLES D'ERREURS SUR LE JEU DE TEST")
print("=" * 60)


if len(errors) == 0:

    print(
        "Aucune erreur !"
    )

else:

    for i, error in enumerate(
        errors[:20],
        start=1
    ):

        print()
        print(
            "Erreur",
            i
        )


        print(
            "Ajami      :",
            error["Ajami"]
        )


        print(
            "Attendu    :",
            error["Wolof_attendu"]
        )


        print(
            "Prediction :",
            error["Wolof_predit"]
        )


# ============================================================
# 18. SAUVEGARDE DES ERREURS
# ============================================================

ERROR_FILE = (
    MODEL_DIR
    / "test_errors_reverse.csv"
)


errors_df = pd.DataFrame(
    errors
)


errors_df.to_csv(
    ERROR_FILE,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 60)
print("SAUVEGARDE")
print("=" * 60)


print(
    "Erreurs du jeu de test sauvegardées dans :"
)


print(
    ERROR_FILE
)


# ============================================================
# 19. FIN
# ============================================================

print()
print("=" * 60)
print("EVALUATION DU TEST REVERSE TERMINEE")
print("=" * 60)