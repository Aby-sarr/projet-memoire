import torch
import json
import pandas as pd

from collections import Counter

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

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 60)
print("ANALYSE DES ERREURS DU JEU DE TEST")
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

ajami_itos = ajami_vocab["itos"]


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
    len(ajami_itos)
)


# ============================================================
# 3. CONSTRUCTION DU MODELE
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


print(
    "Modèle créé avec succès !"
)


# ============================================================
# 4. CHARGEMENT DU MEILLEUR MODELE
# ============================================================

MODEL_PATH = (
    MODEL_DIR / "best_model.pt"
)


print()
print("=" * 60)
print("CHARGEMENT DU MEILLEUR MODELE")
print("=" * 60)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print(
    "Modèle chargé avec succès !"
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
# 5. CHARGEMENT DU CORPUS
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
# 6. SEPARATION TRAIN / VALIDATION / TEST
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


total_corpus = len(df)


train_size = int(
    0.80 * total_corpus
)


valid_size = int(
    0.10 * total_corpus
)


test_size = (
    total_corpus
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


# ============================================================
# 7. CREATION DU JEU DE TEST
# ============================================================

test_df = df.iloc[
    test_indices
].reset_index(
    drop=True
)


print()
print(
    "Nombre de phrases utilisées pour l'analyse :",
    len(test_df)
)


# ============================================================
# 8. FONCTION DE PREDICTION
# ============================================================

def predict(sentence):
    """
    Transforme une phrase Wolof Latin
    en Wolof Ajami.
    """

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


    source_indices = [
        wolof_stoi["<SOS>"]
    ]

    source_indices.extend(
        tokens
    )

    source_indices.append(
        wolof_stoi["<EOS>"]
    )


    source = torch.tensor(
        source_indices,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)


    MAX_LENGTH = 100


    pad_idx = ajami_itos.index(
        "<PAD>"
    )


    sos_idx = ajami_itos.index(
        "<SOS>"
    )


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


        if (
            indice < 0
            or indice >= len(ajami_itos)
        ):

            continue


        caractere = ajami_itos[
            indice
        ]


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


    return "".join(
        resultat
    )


# ============================================================
# 9. ALIGNEMENT DES ERREURS
# ============================================================

def analyze_character_errors(expected, predicted):
    """
    Compare deux séquences de caractères et calcule :

    - insertions
    - suppressions
    - substitutions

    L'analyse est réalisée avec une distance d'édition
    de type Levenshtein.
    """

    n = len(expected)
    m = len(predicted)


    # --------------------------------------------------------
    # Matrice de distance
    # --------------------------------------------------------

    dp = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]


    for i in range(n + 1):
        dp[i][0] = i


    for j in range(m + 1):
        dp[0][j] = j


    # --------------------------------------------------------
    # Calcul de la distance
    # --------------------------------------------------------

    for i in range(1, n + 1):

        for j in range(1, m + 1):

            if expected[i - 1] == predicted[j - 1]:

                cost = 0

            else:

                cost = 1


            deletion = (
                dp[i - 1][j] + 1
            )


            insertion = (
                dp[i][j - 1] + 1
            )


            substitution = (
                dp[i - 1][j - 1]
                + cost
            )


            dp[i][j] = min(
                deletion,
                insertion,
                substitution
            )


    # --------------------------------------------------------
    # Retour arrière pour déterminer le type d'erreur
    # --------------------------------------------------------

    i = n
    j = m


    insertions = 0
    suppressions = 0
    substitutions = 0


    while i > 0 or j > 0:

        # Cas identique
        if (
            i > 0
            and j > 0
            and expected[i - 1] == predicted[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):

            i -= 1
            j -= 1

            continue


        # Substitution
        if (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + 1
        ):

            substitutions += 1

            i -= 1
            j -= 1

            continue


        # Suppression
        if (
            i > 0
            and dp[i][j] == dp[i - 1][j] + 1
        ):

            suppressions += 1

            i -= 1

            continue


        # Insertion
        if (
            j > 0
            and dp[i][j] == dp[i][j - 1] + 1
        ):

            insertions += 1

            j -= 1

            continue


        # Sécurité
        break


    return {
        "insertions": insertions,
        "suppressions": suppressions,
        "substitutions": substitutions,
        "distance": dp[n][m]
    }


# ============================================================
# 10. ANALYSE DES ERREURS
# ============================================================

print()
print("=" * 60)
print("ANALYSE DES ERREURS DU JEU DE TEST")
print("=" * 60)


total = 0

correct = 0


errors = []


# Fréquences caractères
wolof_characters = Counter()

ajami_expected_characters = Counter()

ajami_predicted_characters = Counter()


# Totaux des erreurs
total_insertions = 0
total_suppressions = 0
total_substitutions = 0
total_edit_distance = 0


# ------------------------------------------------------------
# Analyse phrase par phrase
# ------------------------------------------------------------

for index, row in test_df.iterrows():

    wolof = str(
        row["Wolof"]
    )


    attendu = str(
        row["ajami"]
    )


    prediction = predict(
        wolof
    )


    total += 1


    # --------------------------------------------------------
    # Phrase correcte
    # --------------------------------------------------------

    if prediction == attendu:

        correct += 1


    # --------------------------------------------------------
    # Phrase incorrecte
    # --------------------------------------------------------

    else:

        error_info = analyze_character_errors(
            attendu,
            prediction
        )


        total_insertions += (
            error_info["insertions"]
        )


        total_suppressions += (
            error_info["suppressions"]
        )


        total_substitutions += (
            error_info["substitutions"]
        )


        total_edit_distance += (
            error_info["distance"]
        )


        errors.append(
            {
                "Wolof": wolof,
                "Ajami_attendu": attendu,
                "Ajami_predit": prediction,
                "Insertions": error_info["insertions"],
                "Suppressions": error_info["suppressions"],
                "Substitutions": error_info["substitutions"],
                "Distance_edition": error_info["distance"]
            }
        )


    # --------------------------------------------------------
    # Fréquence caractères Wolof
    # --------------------------------------------------------

    for caractere in wolof:

        wolof_characters[
            caractere
        ] += 1


    # --------------------------------------------------------
    # Fréquence caractères Ajami attendus
    # --------------------------------------------------------

    for caractere in attendu:

        ajami_expected_characters[
            caractere
        ] += 1


    # --------------------------------------------------------
    # Fréquence caractères Ajami prédits
    # --------------------------------------------------------

    for caractere in prediction:

        ajami_predicted_characters[
            caractere
        ] += 1


    # --------------------------------------------------------
    # Progression
    # --------------------------------------------------------

    if total % 100 == 0:

        print(
            f"Analyse : "
            f"{total}/{len(test_df)} phrases..."
        )


# ============================================================
# 11. RESULTATS GENERAUX
# ============================================================

accuracy = (
    correct / total
) * 100


incorrect = (
    total - correct
)


print()
print("=" * 60)
print("RESULTATS GENERAUX - JEU DE TEST")
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


# ============================================================
# 12. TYPOLOGIE DES ERREURS
# ============================================================

print()
print("=" * 60)
print("TYPOLOGIE DES ERREURS")
print("=" * 60)


print(
    "Insertions de caractères :",
    total_insertions
)


print(
    "Suppressions de caractères :",
    total_suppressions
)


print(
    "Substitutions de caractères :",
    total_substitutions
)


print(
    "Distance d'édition totale :",
    total_edit_distance
)


# ============================================================
# 13. FREQUENCE DES CARACTERES WOLOF
# ============================================================

print()
print("=" * 60)
print("FREQUENCE DES CARACTERES WOLOF")
print("=" * 60)


for caractere, nombre in (
    wolof_characters.most_common()
):

    print(
        f"{repr(caractere)} : {nombre}"
    )


# ============================================================
# 14. FREQUENCE DES CARACTERES AJAMI ATTENDUS
# ============================================================

print()
print("=" * 60)
print("FREQUENCE DES CARACTERES AJAMI ATTENDUS")
print("=" * 60)


for caractere, nombre in (
    ajami_expected_characters.most_common()
):

    print(
        f"{repr(caractere)} : {nombre}"
    )


# ============================================================
# 15. FREQUENCE DES CARACTERES AJAMI PREDITS
# ============================================================

print()
print("=" * 60)
print("FREQUENCE DES CARACTERES AJAMI PREDITS")
print("=" * 60)


for caractere, nombre in (
    ajami_predicted_characters.most_common()
):

    print(
        f"{repr(caractere)} : {nombre}"
    )


# ============================================================
# 16. EXEMPLES D'ERREURS
# ============================================================

print()
print("=" * 60)
print("EXEMPLES D'ERREURS")
print("=" * 60)


nombre_exemples = min(
    30,
    len(errors)
)


for i in range(
    nombre_exemples
):

    erreur = errors[i]


    print()
    print(
        f"Erreur {i + 1}"
    )


    print(
        "Wolof      :",
        erreur["Wolof"]
    )


    print(
        "Attendu    :",
        erreur["Ajami_attendu"]
    )


    print(
        "Prediction :",
        erreur["Ajami_predit"]
    )


    print(
        "Insertions :",
        erreur["Insertions"]
    )


    print(
        "Suppressions :",
        erreur["Suppressions"]
    )


    print(
        "Substitutions :",
        erreur["Substitutions"]
    )


# ============================================================
# 17. SAUVEGARDE DES ERREURS
# ============================================================

ERROR_FILE = (
    MODEL_DIR
    / "errors_analysis.csv"
)


errors_df = pd.DataFrame(
    errors
)


errors_df.to_csv(
    ERROR_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. SAUVEGARDE DES FREQUENCES
# ============================================================

FREQUENCY_FILE = (
    MODEL_DIR
    / "character_frequencies_test.csv"
)


all_characters = sorted(
    set(
        list(wolof_characters.keys())
        + list(ajami_expected_characters.keys())
        + list(ajami_predicted_characters.keys())
    )
)


frequency_rows = []


for caractere in all_characters:

    frequency_rows.append(
        {
            "Caracter": caractere,
            "Wolof": wolof_characters.get(
                caractere,
                0
            ),
            "Ajami_attendu": ajami_expected_characters.get(
                caractere,
                0
            ),
            "Ajami_predit": ajami_predicted_characters.get(
                caractere,
                0
            )
        }
    )


frequency_df = pd.DataFrame(
    frequency_rows
)


frequency_df.to_csv(
    FREQUENCY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 19. SAUVEGARDE DU RESUME
# ============================================================

SUMMARY_FILE = (
    MODEL_DIR
    / "analysis_summary.csv"
)


summary_df = pd.DataFrame(
    [
        {
            "seed": SEED,
            "total_test": total,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": accuracy,
            "best_epoch": checkpoint.get(
                "epoch",
                ""
            ),
            "validation_loss": checkpoint.get(
                "valid_loss",
                ""
            ),
            "insertions": total_insertions,
            "suppressions": total_suppressions,
            "substitutions": total_substitutions,
            "edit_distance_total": total_edit_distance
        }
    ]
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. AFFICHAGE DES FICHIERS
# ============================================================

print()
print("=" * 60)
print("SAUVEGARDE")
print("=" * 60)


print(
    "Analyse des erreurs :"
)

print(
    ERROR_FILE
)


print()
print(
    "Fréquences des caractères :"
)

print(
    FREQUENCY_FILE
)


print()
print(
    "Résumé de l'analyse :"
)

print(
    SUMMARY_FILE
)


# ============================================================
# 21. FIN
# ============================================================

print()
print("=" * 60)
print("ANALYSE DU JEU DE TEST TERMINEE")
print("=" * 60)