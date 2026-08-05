import pandas as pd
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher


INPUT_FILE = Path("saved_models/errors_analysis_reverse.csv")
OUTPUT_FILE = Path("saved_models/substitution_analysis_reverse.csv")


def extract_substitutions(expected, predicted):

    matcher = SequenceMatcher(
        None,
        expected,
        predicted
    )

    substitutions = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == "replace":

            expected_part = expected[i1:i2]
            predicted_part = predicted[j1:j2]

            # Cas caractère contre caractère
            if (
                len(expected_part) == 1
                and len(predicted_part) == 1
            ):
                substitutions.append(
                    (
                        expected_part,
                        predicted_part
                    )
                )

            # Cas de plusieurs caractères
            else:

                sub_matcher = SequenceMatcher(
                    None,
                    expected_part,
                    predicted_part
                )

                for (
                    sub_tag,
                    a1,
                    a2,
                    b1,
                    b2
                ) in sub_matcher.get_opcodes():

                    if sub_tag == "replace":

                        a = expected_part[a1:a2]
                        b = predicted_part[b1:b2]

                        if (
                            len(a) == 1
                            and len(b) == 1
                        ):
                            substitutions.append(
                                (a, b)
                            )

    return substitutions


print("=" * 60)
print("ANALYSE DES SUBSTITUTIONS - AJAMI -> WOLOF LATIN")
print("=" * 60)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Fichier introuvable : {INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)


print()
print(
    "Nombre de lignes analysées :",
    len(df)
)


all_substitutions = []


for _, row in df.iterrows():

    expected = str(
        row["Wolof_attendu"]
    )

    predicted = str(
        row["Wolof_predit"]
    )

    substitutions = extract_substitutions(
        expected,
        predicted
    )

    all_substitutions.extend(
        substitutions
    )


counter = Counter(
    all_substitutions
)


rows = []


for (expected, predicted), count in counter.most_common():

    rows.append({

        "caractere_attendu": expected,

        "caractere_predit": predicted,

        "nombre": count

    })


result = pd.DataFrame(
    rows
)


if len(result) > 0:

    total = result["nombre"].sum()

    result["pourcentage"] = (
        result["nombre"]
        / total
        * 100
    ).round(2)

else:

    total = 0

    result["pourcentage"] = []


result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 60)
print("RESULTATS")
print("=" * 60)

print()

print(
    "Nombre total de substitutions caractère-à-caractère :",
    total
)

print()

if len(result) > 0:

    for _, row in result.head(30).iterrows():

        print(
            f"{row['caractere_attendu']!r} -> "
            f"{row['caractere_predit']!r} : "
            f"{int(row['nombre'])} "
            f"({row['pourcentage']:.2f} %)"
        )


print()
print("=" * 60)

print(
    "Analyse complète sauvegardée dans :"
)

print(
    OUTPUT_FILE
)

print("=" * 60)