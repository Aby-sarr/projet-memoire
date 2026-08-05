import pandas as pd
from pathlib import Path
from collections import Counter


INPUT_FILE = Path("saved_models/test_errors_reverse.csv")
OUTPUT_FILE = Path("saved_models/confusions_words_reverse.csv")


TARGET_PAIRS = {
    ("o", "u"),
    ("u", "o"),
    ("i", "y"),
    ("y", "i"),
}


def extract_word(text, position):
    """
    Retourne le mot contenant le caractère en position donnée.
    """
    if position >= len(text):
        return ""

    start = position
    end = position

    while start > 0 and not text[start - 1].isspace():
        start -= 1

    while end < len(text) and not text[end].isspace():
        end += 1

    return text[start:end]


def analyze_pair(expected, predicted):
    """
    Compare deux séquences caractère par caractère.
    """
    results = []

    max_len = min(len(expected), len(predicted))

    for i in range(max_len):

        e = expected[i]
        p = predicted[i]

        if (e, p) not in TARGET_PAIRS:
            continue

        word_expected = extract_word(expected, i)
        word_predicted = extract_word(predicted, i)

        # Position du caractère dans le mot
        prefix = expected[:i]

        last_space = prefix.rfind(" ")

        if last_space == -1:
            word_position = i
        else:
            word_position = i - last_space - 1

        if word_position == 0:
            position_type = "Début du mot"
        elif i + 1 < len(expected) and expected[i + 1].isspace():
            position_type = "Fin du mot"
        else:
            position_type = "Milieu du mot"

        results.append({
            "confusion": f"{e} -> {p}",
            "attendu_caractere": e,
            "predit_caractere": p,
            "mot_attendu": word_expected,
            "mot_predit": word_predicted,
            "position_dans_mot": position_type,
            "position_caractere": word_position,
            "phrase_attendue": expected,
            "phrase_predite": predicted
        })

    return results


# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 60)
print("ANALYSE DETAILLEE DES CONFUSIONS")
print("AJAMI -> WOLOF LATIN")
print("=" * 60)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

required_columns = [
    "Ajami",
    "Wolof_attendu",
    "Wolof_predit"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Colonne introuvable : {column}"
        )

print()
print("Nombre de lignes :", len(df))


# ============================================================
# ANALYSE
# ============================================================

all_results = []

for _, row in df.iterrows():

    expected = str(row["Wolof_attendu"])
    predicted = str(row["Wolof_predit"])

    results = analyze_pair(
        expected,
        predicted
    )

    for result in results:
        result["Ajami"] = row["Ajami"]
        all_results.append(result)


result_df = pd.DataFrame(all_results)


# ============================================================
# SAUVEGARDE
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# STATISTIQUES GENERALES
# ============================================================

print()
print("=" * 60)
print("RESULTATS GENERAUX")
print("=" * 60)

print()

print(
    "Nombre total de confusions :",
    len(result_df)
)


# ============================================================
# PAR TYPE DE CONFUSION
# ============================================================

print()
print("=" * 60)
print("CONFUSIONS")
print("=" * 60)

if len(result_df) > 0:

    confusion_counts = (
        result_df["confusion"]
        .value_counts()
    )

    for confusion, count in confusion_counts.items():

        percentage = (
            count / len(result_df) * 100
        )

        print(
            f"{confusion:10s} "
            f"{count:4d} "
            f"({percentage:6.2f} %)"
        )


# ============================================================
# MOTS LES PLUS FREQUENTS
# ============================================================

print()
print("=" * 60)
print("MOTS LES PLUS FREQUEMMENT CONCERNES")
print("=" * 60)

if len(result_df) > 0:

    word_counts = (
        result_df["mot_attendu"]
        .value_counts()
        .head(50)
    )

    for word, count in word_counts.items():

        print(
            f"{word:30s} : {count}"
        )


# ============================================================
# POSITION DANS LE MOT
# ============================================================

print()
print("=" * 60)
print("POSITION DES CONFUSIONS")
print("=" * 60)

if len(result_df) > 0:

    position_counts = (
        result_df["position_dans_mot"]
        .value_counts()
    )

    for position, count in position_counts.items():

        percentage = (
            count / len(result_df) * 100
        )

        print(
            f"{position:20s} "
            f"{count:4d} "
            f"({percentage:6.2f} %)"
        )


# ============================================================
# ANALYSE PAR CONFUSION ET POSITION
# ============================================================

print()
print("=" * 60)
print("CONFUSION + POSITION")
print("=" * 60)

if len(result_df) > 0:

    grouped = (
        result_df
        .groupby(
            ["confusion", "position_dans_mot"]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    for (
        confusion,
        position
    ), count in grouped.items():

        print(
            f"{confusion:8s} | "
            f"{position:20s} : "
            f"{count}"
        )


# ============================================================
# EXEMPLES PAR CONFUSION
# ============================================================

print()
print("=" * 60)
print("EXEMPLES PAR TYPE DE CONFUSION")
print("=" * 60)

for confusion in [
    "o -> u",
    "u -> o",
    "i -> y",
    "y -> i"
]:

    subset = result_df[
        result_df["confusion"] == confusion
    ]

    print()
    print("-" * 60)
    print(confusion)
    print("-" * 60)

    for _, row in subset.head(15).iterrows():

        print()
        print(
            "Mot attendu :",
            row["mot_attendu"]
        )

        print(
            "Mot prédit  :",
            row["mot_predit"]
        )

        print(
            "Position    :",
            row["position_dans_mot"]
        )

        print(
            "Phrase      :",
            row["phrase_attendue"]
        )

        print(
            "Prédiction  :",
            row["phrase_predite"]
        )


# ============================================================
# FIN
# ============================================================

print()
print("=" * 60)
print("ANALYSE TERMINEE")
print("=" * 60)

print()
print(
    "Fichier sauvegardé :",
    OUTPUT_FILE
)