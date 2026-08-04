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

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 60)
print("EVALUATION FINALE SUR LE JEU DE TEST")
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

with open(WOLOF_VOCAB_FILE, "r", encoding="utf-8") as f:
    wolof_vocab = json.load(f)

with open(AJAMI_VOCAB_FILE, "r", encoding="utf-8") as f:
    ajami_vocab = json.load(f)


wolof_stoi = wolof_vocab["stoi"]
wolof_itos = wolof_vocab["itos"]

ajami_stoi = ajami_vocab["stoi"]
ajami_itos = ajami_vocab["itos"]


# Conversion éventuelle de itos
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
# 3. VERIFICATION DES TOKENS
# ============================================================

print()
print("=" * 60)
print("VERIFICATION DES TOKENS SPECIAUX")
print("=" * 60)

for token in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]:

    if token not in wolof_stoi:
        raise ValueError(
            f"Token Wolof manquant : {token}"
        )

for token in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]:

    if token not in ajami_stoi:
        raise ValueError(
            f"Token Ajami manquant : {token}"
        )

print("Tokens spéciaux Wolof : OK")
print("Tokens spéciaux Ajami : OK")


# ============================================================
# 4. CONSTRUCTION DU MODELE
# ============================================================

print()
print("=" * 60)
print("CONSTRUCTION DU MODELE")
print("=" * 60)

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

print("Modèle créé avec succès !")


# ============================================================
# 5. CHARGEMENT DU MEILLEUR MODELE
# ============================================================

print()
print("=" * 60)
print("CHARGEMENT DU MEILLEUR MODELE")
print("=" * 60)

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

df = df.reset_index(drop=True)

print(
    "Nombre total de phrases :",
    len(df)
)


# ============================================================
# 7. SEPARATION TRAIN / VALIDATION / TEST
# ============================================================

print()
print("=" * 60)
print("SEPARATION TRAIN / VALIDATION / TEST")
print("=" * 60)

generator = torch.Generator().manual_seed(SEED)

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

train_indices = indices[:train_size]

valid_indices = indices[
    train_size:train_size + valid_size
]

test_indices = indices[
    train_size + valid_size:
]

print("Train :", len(train_indices))
print("Validation :", len(valid_indices))
print("Test :", len(test_indices))


test_df = df.iloc[
    test_indices
].reset_index(drop=True)

print()
print(
    "Nombre de phrases utilisées pour le test :",
    len(test_df)
)


# ============================================================
# 8. FONCTION DE PREDICTION
# ============================================================

def predict(sentence):

    tokens = []

    for caractere in sentence:

        tokens.append(
            wolof_stoi.get(
                caractere,
                wolof_stoi["<UNK>"]
            )
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

    MAX_LENGTH = 100

    pad_idx = ajami_stoi["<PAD>"]
    sos_idx = ajami_stoi["<SOS>"]

    target = torch.full(
        (1, MAX_LENGTH),
        pad_idx,
        dtype=torch.long,
        device=DEVICE
    )

    target[0, 0] = sos_idx

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

        indice = indice.item()

        if indice not in ajami_itos:
            continue

        caractere = ajami_itos[indice]

        if caractere == "<EOS>":
            break

        if caractere in [
            "<PAD>",
            "<SOS>"
        ]:
            continue

        resultat.append(
            caractere
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

            if reference[i - 1] == hypothesis[j - 1]:

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
print("=" * 60)

correct = 0
total = 0

total_char_errors = 0
total_reference_chars = 0

total_word_errors = 0
total_reference_words = 0

errors = []


for index, row in test_df.iterrows():

    wolof = str(row["Wolof"])
    expected = str(row["ajami"])

    prediction = predict(
        wolof
    )


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    if prediction == expected:

        correct += 1


    # --------------------------------------------------------
    # CER
    # --------------------------------------------------------

    char_distance = levenshtein_distance(
        expected,
        prediction
    )

    total_char_errors += char_distance

    total_reference_chars += len(
        expected
    )


    # --------------------------------------------------------
    # WER
    # --------------------------------------------------------

    reference_words = expected.split()
    prediction_words = prediction.split()

    word_distance = levenshtein_distance(
        reference_words,
        prediction_words
    )

    total_word_errors += word_distance

    total_reference_words += len(
        reference_words
    )


    # --------------------------------------------------------
    # Erreurs
    # --------------------------------------------------------

    if prediction != expected:

        errors.append({

            "Wolof": wolof,

            "Ajami_attendu": expected,

            "Ajami_predit": prediction

        })


    total += 1


    # --------------------------------------------------------
    # Progression
    # --------------------------------------------------------

    if total % 100 == 0:

        print(
            f"Évaluation : "
            f"{total}/{len(test_df)} phrases..."
        )


# ============================================================
# 11. CALCUL DES METRIQUES
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


incorrect = total - correct


# ============================================================
# 12. RESULTATS FINAUX
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
# 13. EXEMPLES D'ERREURS
# ============================================================

print()
print("=" * 60)
print("EXEMPLES D'ERREURS SUR LE JEU DE TEST")
print("=" * 60)

if len(errors) == 0:

    print("Aucune erreur !")

else:

    for i, error in enumerate(
        errors[:20],
        start=1
    ):

        print()
        print("Erreur", i)

        print(
            "Wolof      :",
            error["Wolof"]
        )

        print(
            "Attendu    :",
            error["Ajami_attendu"]
        )

        print(
            "Prediction :",
            error["Ajami_predit"]
        )


# ============================================================
# 14. SAUVEGARDE DES ERREURS
# ============================================================

ERROR_FILE = (
    MODEL_DIR
    / "test_errors.csv"
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
# 15. FIN
# ============================================================

print()
print("=" * 60)
print("EVALUATION DU TEST TERMINEE")
print("=" * 60)