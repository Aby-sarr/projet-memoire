import pandas as pd
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

ERROR_FILE = (
    Path("saved_models")
    / "test_errors_reverse.csv"
)

DETAIL_OUTPUT = (
    Path("saved_models")
    / "errors_analysis_reverse.csv"
)

SUMMARY_OUTPUT = (
    Path("saved_models")
    / "error_categories_reverse.csv"
)


# ============================================================
# NORMALISATION
# ============================================================

def remove_accents(text):

    text = unicodedata.normalize(
        "NFD",
        text
    )

    return "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )


def remove_punctuation(text):

    return re.sub(
        r"[^\w\s]",
        "",
        text,
        flags=re.UNICODE
    )


def normalize_spaces(text):

    return " ".join(
        text.split()
    )


# ============================================================
# OPERATIONS D'EDITION
# ============================================================

def get_operations(expected, predicted):

    matcher = SequenceMatcher(
        None,
        expected,
        predicted
    )

    operations = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == "equal":
            continue

        operations.append({
            "operation": tag,
            "expected": expected[i1:i2],
            "predicted": predicted[j1:j2]
        })

    return operations


# ============================================================
# DETECTION DUPLICATION
# ============================================================

def contains_duplication(expected, predicted):

    # Caractère répété dans la prédiction
    if re.search(
        r"(.)\1",
        predicted
    ):

        if not re.search(
            r"(.)\1",
            expected
        ):

            return True


    # Séquences répétées
    for size in range(2, 5):

        pattern = rf"(.{{{size}}})\1"

        if (
            re.search(
                pattern,
                predicted
            )
            and
            not re.search(
                pattern,
                expected
            )
        ):

            return True


    return False


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_error(expected, predicted):

    # --------------------------------------------------------
    # 1. MAJUSCULE / MINUSCULE
    # --------------------------------------------------------

    if expected.casefold() == predicted.casefold():

        return "Majuscule / minuscule"


    # --------------------------------------------------------
    # 2. DIACRITIQUE / ACCENT
    # --------------------------------------------------------

    if (
        remove_accents(expected).casefold()
        ==
        remove_accents(predicted).casefold()
    ):

        return "Diacritique / accent"


    # --------------------------------------------------------
    # 3. PONCTUATION
    # --------------------------------------------------------

    if (
        normalize_spaces(
            remove_punctuation(expected)
        )
        ==
        normalize_spaces(
            remove_punctuation(predicted)
        )
    ):

        return "Ponctuation"


    # --------------------------------------------------------
    # 4. DUPLICATION
    # --------------------------------------------------------

    if contains_duplication(
        expected,
        predicted
    ):

        return "Duplication / répétition"


    # --------------------------------------------------------
    # 5. OPERATIONS
    # --------------------------------------------------------

    operations = get_operations(
        expected,
        predicted
    )


    if len(operations) == 0:

        return "Autre"


    operation_types = [
        op["operation"]
        for op in operations
    ]


    # --------------------------------------------------------
    # INSERTION
    # --------------------------------------------------------

    if all(
        op == "insert"
        for op in operation_types
    ):

        return "Insertion"


    # --------------------------------------------------------
    # SUPPRESSION
    # --------------------------------------------------------

    if all(
        op == "delete"
        for op in operation_types
    ):

        return "Suppression"


    # --------------------------------------------------------
    # SUBSTITUTION
    # --------------------------------------------------------

    if all(
        op == "replace"
        for op in operation_types
    ):

        return "Substitution"


    # --------------------------------------------------------
    # COMBINAISON
    # --------------------------------------------------------

    return "Erreur multiple / combinaison"


# ============================================================
# DEBUT
# ============================================================

print("=" * 60)
print("ANALYSE DES ERREURS - AJAMI -> WOLOF LATIN")
print("=" * 60)


print()
print(
    "Fichier :",
    ERROR_FILE
)


# ============================================================
# VERIFICATION FICHIER
# ============================================================

if not ERROR_FILE.exists():

    print()
    print("ERREUR : fichier introuvable.")

    print(
        "Vérifie que test_errors_reverse.csv se trouve dans :"
    )

    print(
        "saved_models/"
    )

    raise SystemExit


# ============================================================
# CHARGEMENT
# ============================================================

df = pd.read_csv(
    ERROR_FILE
)


print()
print(
    "Nombre d'erreurs chargées :",
    len(df)
)


# ============================================================
# COLONNES
# ============================================================

print()
print("Colonnes détectées :")

for column in df.columns:

    print(
        " -",
        column
    )


# ============================================================
# COLONNES REVERSE
# ============================================================

source_column = "Ajami"

expected_column = "Wolof_attendu"

predicted_column = "Wolof_predit"


if source_column not in df.columns:

    raise ValueError(
        f"Colonne introuvable : {source_column}"
    )


if expected_column not in df.columns:

    raise ValueError(
        f"Colonne introuvable : {expected_column}"
    )


if predicted_column not in df.columns:

    raise ValueError(
        f"Colonne introuvable : {predicted_column}"
    )


print()
print(
    "Colonne source Ajami :",
    source_column
)

print(
    "Colonne attendue Wolof :",
    expected_column
)

print(
    "Colonne prédite Wolof :",
    predicted_column
)


# ============================================================
# ANALYSE
# ============================================================

results = []


for index, row in df.iterrows():

    source = str(
        row[source_column]
    )

    expected = str(
        row[expected_column]
    )

    predicted = str(
        row[predicted_column]
    )


    category = classify_error(
        expected,
        predicted
    )


    operations = get_operations(
        expected,
        predicted
    )


    operation_string = " | ".join(
        [
            op["operation"]
            for op in operations
        ]
    )


    results.append({

        "id": index + 1,

        "Ajami": source,

        "Wolof_attendu": expected,

        "Wolof_predit": predicted,

        "category": category,

        "operations": operation_string

    })


analysis_df = pd.DataFrame(
    results
)


# ============================================================
# SAUVEGARDE DETAIL
# ============================================================

analysis_df.to_csv(
    DETAIL_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# STATISTIQUES
# ============================================================

total_errors = len(
    analysis_df
)


summary = (
    analysis_df[
        "category"
    ]
    .value_counts()
    .reset_index()
)


summary.columns = [
    "category",
    "count"
]


summary["percentage"] = (
    summary["count"]
    / total_errors
    * 100
)


summary["percentage"] = (
    summary["percentage"]
    .round(2)
)


# ============================================================
# ORDRE DES CATEGORIES
# ============================================================

category_order = [

    "Majuscule / minuscule",

    "Diacritique / accent",

    "Ponctuation",

    "Duplication / répétition",

    "Insertion",

    "Suppression",

    "Substitution",

    "Erreur multiple / combinaison",

    "Autre"

]


summary["order"] = (
    summary["category"]
    .apply(
        lambda x:
        category_order.index(x)
        if x in category_order
        else 999
    )
)


summary = (
    summary
    .sort_values("order")
    .drop(
        columns=["order"]
    )
)


# ============================================================
# SAUVEGARDE RESUME
# ============================================================

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# AFFICHAGE
# ============================================================

print()
print("=" * 60)
print("RESULTATS DE L'ANALYSE")
print("=" * 60)

print()


for _, row in summary.iterrows():

    print(
        f"{row['category']:<35} "
        f"{int(row['count']):>4} erreurs "
        f"({row['percentage']:.2f} %)"
    )


print()
print("-" * 60)

print(
    f"TOTAL : {total_errors} erreurs"
)

print(
    "TOTAL : 100.00 %"
)


print()
print("=" * 60)
print("FICHIERS SAUVEGARDES")
print("=" * 60)

print()

print(
    "Détail :",
    DETAIL_OUTPUT
)

print(
    "Résumé :",
    SUMMARY_OUTPUT
)


print()
print("=" * 60)
print("ANALYSE TERMINEE")
print("=" * 60)