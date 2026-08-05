import pandas as pd
from pathlib import Path
from collections import Counter


INPUT_FILE = Path("saved_models/test_errors_reverse.csv")
OUTPUT_FILE = Path("saved_models/confusions_context_reverse.csv")


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_PAIRS = {
    ("o", "u"),
    ("u", "o"),
    ("i", "y"),
    ("y", "i"),
}


# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 60)
print("ANALYSE CONTEXTUELLE DES CONFUSIONS")
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

print()
print("Nombre de lignes :", len(df))

required = [
    "Ajami",
    "Wolof_attendu",
    "Wolof_predit"
]

for column in required:
    if column not in df.columns:
        raise ValueError(
            f"Colonne introuvable : {column}"
        )


# ============================================================
# EXTRACTION DES SUBSTITUTIONS
# ============================================================

def find_confusions(expected, predicted):

    results = []

    min_len = min(
        len(expected),
        len(predicted)
    )

    for i in range(min_len):

        expected_char = expected[i]
        predicted_char = predicted[i]

        pair = (
            expected_char,
            predicted_char
        )

        if pair in TARGET_PAIRS:

            start = max(0, i - 5)
            end = min(
                len(expected),
                i + 6
            )

            context_expected = expected[start:end]

            start_pred = max(0, i - 5)
            end_pred = min(
                len(predicted),
                i + 6
            )

            context_predicted = predicted[
                start_pred:end_pred
            ]

            results.append({
                "attendu": expected_char,
                "predit": predicted_char,
                "position": i,
                "contexte_attendu": context_expected,
                "contexte_predit": context_predicted
            })

    return results


# ============================================================
# ANALYSE
# ============================================================

all_results = []

for _, row in df.iterrows():

    expected = str(
        row["Wolof_attendu"]
    )

    predicted = str(
        row["Wolof_predit"]
    )

    confusions = find_confusions(
        expected,
        predicted
    )

    for item in confusions:

        item["Ajami"] = row["Ajami"]
        item["Wolof_attendu"] = expected
        item["Wolof_predit"] = predicted

        all_results.append(item)


result = pd.DataFrame(
    all_results
)


# ============================================================
# SAUVEGARDE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# STATISTIQUES
# ============================================================

print()
print("=" * 60)
print("RESULTATS")
print("=" * 60)

print()

print(
    "Nombre total de confusions analysées :",
    len(result)
)

print()

if len(result) > 0:

    counter = Counter(
        zip(
            result["attendu"],
            result["predit"]
        )
    )

    for pair, count in counter.most_common():

        expected_char, predicted_char = pair

        print(
            f"{expected_char!r} -> "
            f"{predicted_char!r} : "
            f"{count}"
        )


# ============================================================
# EXEMPLES
# ============================================================

print()
print("=" * 60)
print("EXEMPLES DE CONTEXTES")
print("=" * 60)

if len(result) > 0:

    for _, row in result.head(50).iterrows():

        print()
        print(
            f"{row['attendu']!r} -> "
            f"{row['predit']!r}"
        )

        print(
            "Attendu :",
            row["Wolof_attendu"]
        )

        print(
            "Prédit  :",
            row["Wolof_predit"]
        )

        print(
            "Contexte attendu :",
            row["contexte_attendu"]
        )

        print(
            "Contexte prédit  :",
            row["contexte_predit"]
        )


print()
print("=" * 60)

print(
    "Fichier sauvegardé :"
)

print(
    OUTPUT_FILE
)

print("=" * 60)