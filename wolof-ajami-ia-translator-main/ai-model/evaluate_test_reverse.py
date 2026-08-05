import torch
import json
import pandas as pd
import random

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


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

MAX_LENGTH = 100

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 60)
print("EVALUATION FINALE - AJAMI -> WOLOF LATIN")
print("=" * 60)

print("Device :", DEVICE)
print("Seed :", SEED)


# ============================================================
# 2. CHARGEMENT DES VOCABULAIRES
# ============================================================

print()
print("=" * 60)
print("CHARGEMENT DES VOCABULAIRES")
print("=" * 60)


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
# Conversion éventuelle de itos
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


required_ajami_tokens = [
    "<PAD>",
    "<SOS>",
    "<EOS>",
    "<UNK>"
]


required_wolof_tokens = [
    "<PAD>",
    "<SOS>",
    "<EOS>",
    "<UNK>"
]


for token in required_ajami_tokens:

    if token not in ajami_stoi:

        raise ValueError(
            f"Token Ajami manquant : {token}"
        )


for token in required_wolof_tokens:

    if token not in wolof_stoi:

        raise ValueError(
            f"Token Wolof manquant : {token}"
        )


print("Tokens spéciaux Ajami : OK")
print("Tokens spéciaux Wolof : OK")


print()
print("Indices des tokens :")

print(
    "Ajami <PAD> =",
    ajami_stoi["<PAD>"]
)

print(
    "Ajami <SOS> =",
    ajami_stoi["<SOS>"]
)

print(
    "Ajami <EOS> =",
    ajami_stoi["<EOS>"]
)

print(
    "Wolof <PAD> =",
    wolof_stoi["<PAD>"]
)

print(
    "Wolof <SOS> =",
    wolof_stoi["<SOS>"]
)

print(
    "Wolof <EOS> =",
    wolof_stoi["<EOS>"]
)


# ============================================================
# 4. CONSTRUCTION DU MODELE REVERSE
# ============================================================

print()
print("=" * 60)
print("CONSTRUCTION DU MODELE AJAMI -> WOLOF")
print("=" * 60)


# ------------------------------------------------------------
# ENCODEUR
# Source = Ajami
# ------------------------------------------------------------

encoder = EncoderGRU(
    input_dim=len(ajami_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)


# ------------------------------------------------------------
# ATTENTION
# ------------------------------------------------------------

attention = BahdanauAttention(
    HIDDEN_SIZE
)


# ------------------------------------------------------------
# DECODEUR
# Cible = Wolof Latin
# ------------------------------------------------------------

decoder = DecoderGRU(
    output_dim=len(wolof_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)


# ------------------------------------------------------------
# SEQ2SEQ
# ------------------------------------------------------------

model = Seq2Seq(
    encoder,
    decoder,
    attention,
    DEVICE
).to(DEVICE)


print(
    "Modèle reverse créé avec succès !"
)


# ============================================================
# 5. CHARGEMENT DU MEILLEUR MODELE
# ============================================================

print()
print("=" * 60)
print("CHARGEMENT DU MEILLEUR MODELE REVERSE")
print("=" * 60)


MODEL_PATH = (
    MODEL_DIR
    / "best_model_reverse.pt"
)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Modèle reverse introuvable : {MODEL_PATH}"
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print(
    "Modèle reverse chargé avec succès !"
)


if "epoch" in checkpoint:

    print(
        "Epoch du meilleur modèle :",
        checkpoint["epoch"]
    )


if "valid_loss" in checkpoint:

    print(
        "Loss validation :",
        checkpoint["valid_loss"]
    )


# ============================================================
# 6. CHARGEMENT DU CORPUS
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
# 7. RECONSTRUCTION EXACTE DU SPLIT
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


total_size = len(df)


train_size = int(
    0.80 * total_size
)


valid_size = int(
    0.10 * total_size
)


test_size = (
    total_size
    - train_size
    - valid_size
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
# 8. PREDICTION AJAMI -> WOLOF
# ============================================================

def predict(sentence):

    sentence = str(sentence)


    # --------------------------------------------------------
    # Encodage de la source Ajami
    # --------------------------------------------------------

    tokens = []


    for caractere in sentence:

        if caractere in ajami_stoi:

            tokens.append(
                ajami_stoi[caractere]
            )

        else:

            tokens.append(
                ajami_stoi["<UNK>"]
            )


    source_indices = (
        [ajami_stoi["<SOS>"]]
        + tokens
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
            model.encoder(source)
        )


    # --------------------------------------------------------
    # PREMIER TOKEN DU DECODEUR
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

    for step in range(MAX_LENGTH):

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
            or predicted_token >= len(wolof_itos)
        ):

            break


        predicted_char = wolof_itos[
            predicted_token
        ]


        # ----------------------------------------------------
        # EOS
        # ----------------------------------------------------

        if predicted_char == "<EOS>":

            break


        # ----------------------------------------------------
        # Ignorer PAD / SOS
        # ----------------------------------------------------

        if predicted_char not in [
            "<PAD>",
            "<SOS>"
        ]:

            resultat.append(
                predicted_char
            )


        # ----------------------------------------------------
        # Le token prédit devient
        # l'entrée suivante
        # ----------------------------------------------------

        input_token = torch.tensor(
            [predicted_token],
            dtype=torch.long,
            device=DEVICE
        )


    return "".join(resultat)


# ============================================================
# 9. DISTANCE DE LEVENSHTEIN
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


    for i in range(1, rows):

        for j in range(1, cols):

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
# 10. EVALUATION
# ============================================================

print()
print("=" * 60)
print("DEBUT DE L'EVALUATION DU JEU DE TEST")
print("DIRECTION : AJAMI -> WOLOF LATIN")
print("=" * 60)


correct = 0

total = 0


total_char_errors = 0

total_reference_chars = 0


total_word_errors = 0

total_reference_words = 0


errors = []


preview_count = 0


# ============================================================
# PARCOURS DU TEST
# ============================================================

for index, row in test_df.iterrows():

    ajami = str(
        row["ajami"]
    )


    expected = str(
        row["Wolof"]
    )


    prediction = predict(
        ajami
    )


    # --------------------------------------------------------
    # Affichage des 10 premiers exemples
    # --------------------------------------------------------

    if preview_count < 10:

        print()
        print(
            f"Exemple {preview_count + 1}"
        )

        print(
            "Ajami      :",
            ajami
        )

        print(
            "Attendu    :",
            expected
        )

        print(
            "Prediction :",
            prediction
        )

        preview_count += 1


    # ========================================================
    # TOTAL
    # ========================================================

    total += 1


    # ========================================================
    # ACCURACY
    # ========================================================

    if prediction == expected:

        correct += 1


    # ========================================================
    # CER
    # ========================================================

    char_distance = (
        levenshtein_distance(
            expected,
            prediction
        )
    )


    total_char_errors += (
        char_distance
    )


    total_reference_chars += (
        len(expected)
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


    word_distance = (
        levenshtein_distance(
            reference_words,
            prediction_words
        )
    )


    total_word_errors += (
        word_distance
    )


    total_reference_words += (
        len(reference_words)
    )


    # ========================================================
    # ENREGISTREMENT DES ERREURS
    # ========================================================

    if prediction != expected:

        errors.append({

            "Ajami":
                ajami,

            "Wolof_attendu":
                expected,

            "Wolof_predit":
                prediction

        })


    # ========================================================
    # PROGRESSION
    # ========================================================

    if total % 100 == 0:

        print(
            f"Évaluation : "
            f"{total}/{len(test_df)} phrases..."
        )


# ============================================================
# 11. CALCUL DES METRIQUES
# ============================================================

if total > 0:

    accuracy = (
        correct / total
    ) * 100

else:

    accuracy = 0.0


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
# 12. RESULTATS FINAUX
# ============================================================

print()
print("=" * 60)
print("RESULTATS FINAUX - AJAMI -> WOLOF LATIN")
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
# 13. EXEMPLES D'ERREURS
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
# 14. SAUVEGARDE DES ERREURS
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
    "Erreurs du jeu de test reverse sauvegardées dans :"
)


print(
    ERROR_FILE
)


# ============================================================
# 15. FIN
# ============================================================

print()
print("=" * 60)
print("EVALUATION REVERSE TERMINEE")
print("=" * 60)


print(
    "Direction : AJAMI -> WOLOF LATIN"
)


print(
    "Modèle : best_model_reverse.pt"
)


print(
    "Test :",
    total,
    "phrases"
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