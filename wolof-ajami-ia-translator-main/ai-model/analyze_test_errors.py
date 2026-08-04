import pandas as pd
from collections import Counter


# ============================================================
# 1. CONFIGURATION
# ============================================================

ERROR_FILE = "saved_models/test_errors.csv"


print("=" * 60)
print("ANALYSE DETAILLEE DES ERREURS DU JEU DE TEST")
print("=" * 60)


# ============================================================
# 2. CHARGEMENT DES ERREURS
# ============================================================

print()
print("=" * 60)
print("CHARGEMENT DU FICHIER DES ERREURS")
print("=" * 60)

df = pd.read_csv(
    ERROR_FILE,
    encoding="utf-8-sig"
)

print(
    "Nombre d'erreurs chargées :",
    len(df)
)


# ============================================================
# 3. VERIFICATION DES COLONNES
# ============================================================

required_columns = [
    "Wolof",
    "Ajami_attendu",
    "Ajami_predit"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Colonne absente : {column}"
        )


# ============================================================
# 4. ANALYSE DES CARACTERES WOLOF
# ============================================================

print()
print("=" * 60)
print("CARACTERES WOLOF PRESENTS DANS LES ERREURS")
print("=" * 60)

wolof_counter = Counter()


for sentence in df["Wolof"].astype(str):

    for character in sentence:

        if character != " ":

            wolof_counter[character] += 1


for character, count in wolof_counter.most_common(30):

    print(
        repr(character),
        ":",
        count
    )


# ============================================================
# 5. CARACTERES AJAMI ATTENDUS DANS LES ERREURS
# ============================================================

print()
print("=" * 60)
print("CARACTERES AJAMI ATTENDUS DANS LES ERREURS")
print("=" * 60)

expected_counter = Counter()


for sentence in df["Ajami_attendu"].astype(str):

    for character in sentence:

        if character != " ":

            expected_counter[character] += 1


for character, count in expected_counter.most_common(30):

    print(
        repr(character),
        ":",
        count
    )


# ============================================================
# 6. CARACTERES AJAMI PREDITS DANS LES ERREURS
# ============================================================

print()
print("=" * 60)
print("CARACTERES AJAMI PREDITS DANS LES ERREURS")
print("=" * 60)

predicted_counter = Counter()


for sentence in df["Ajami_predit"].astype(str):

    for character in sentence:

        if character != " ":

            predicted_counter[character] += 1


for character, count in predicted_counter.most_common(30):

    print(
        repr(character),
        ":",
        count
    )


# ============================================================
# 7. DIFFERENCES ENTRE ATTENDU ET PREDIT
# ============================================================

print()
print("=" * 60)
print("ANALYSE DES DIFFERENCES")
print("=" * 60)


substitutions = Counter()
insertions = Counter()
deletions = Counter()


def levenshtein_operations(reference, hypothesis):

    rows = len(reference) + 1
    cols = len(hypothesis) + 1

    matrix = [
        [0] * cols
        for _ in range(rows)
    ]


    for i in range(rows):

        matrix[i][0] = i


    for j in range(cols):

        matrix[0][j] = j


    for i in range(1, rows):

        for j in range(1, cols):

            if reference[i - 1] == hypothesis[j - 1]:

                cost = 0

            else:

                cost = 1


            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost
            )


    operations = []

    i = rows - 1
    j = cols - 1


    while i > 0 or j > 0:

        # Même caractère
        if (
            i > 0
            and j > 0
            and reference[i - 1] == hypothesis[j - 1]
        ):

            i -= 1
            j -= 1

        # Substitution
        elif (
            i > 0
            and j > 0
            and matrix[i][j]
            == matrix[i - 1][j - 1] + 1
        ):

            operations.append(
                (
                    "SUBSTITUTION",
                    reference[i - 1],
                    hypothesis[j - 1]
                )
            )

            i -= 1
            j -= 1

        # Suppression
        elif (
            i > 0
            and matrix[i][j]
            == matrix[i - 1][j] + 1
        ):

            operations.append(
                (
                    "SUPPRESSION",
                    reference[i - 1],
                    ""
                )
            )

            i -= 1

        # Insertion
        else:

            operations.append(
                (
                    "INSERTION",
                    "",
                    hypothesis[j - 1]
                )
            )

            j -= 1


    return operations


for _, row in df.iterrows():

    expected = str(
        row["Ajami_attendu"]
    )

    predicted = str(
        row["Ajami_predit"]
    )


    operations = levenshtein_operations(
        expected,
        predicted
    )


    for operation, expected_char, predicted_char in operations:

        if operation == "SUBSTITUTION":

            substitutions[
                (
                    expected_char,
                    predicted_char
                )
            ] += 1


        elif operation == "SUPPRESSION":

            deletions[
                expected_char
            ] += 1


        elif operation == "INSERTION":

            insertions[
                predicted_char
            ] += 1


# ============================================================
# 8. CONFUSIONS LES PLUS FREQUENTES
# ============================================================

print()
print("=" * 60)
print("CONFUSIONS DE CARACTERES LES PLUS FREQUENTES")
print("=" * 60)


if len(substitutions) == 0:

    print("Aucune substitution détectée.")

else:

    for (
        (expected_char, predicted_char),
        count
    ) in substitutions.most_common(30):

        print(
            f"Attendu {repr(expected_char)}"
            f" -> Prédit {repr(predicted_char)}"
            f" : {count}"
        )


# ============================================================
# 9. SUPPRESSIONS LES PLUS FREQUENTES
# ============================================================

print()
print("=" * 60)
print("CARACTERES LE PLUS SOUVENT SUPPRIMES")
print("=" * 60)


if len(deletions) == 0:

    print("Aucune suppression détectée.")

else:

    for character, count in deletions.most_common(20):

        print(
            f"{repr(character)} : {count}"
        )


# ============================================================
# 10. INSERTIONS LES PLUS FREQUENTES
# ============================================================

print()
print("=" * 60)
print("CARACTERES LE PLUS SOUVENT AJOUTES")
print("=" * 60)


if len(insertions) == 0:

    print("Aucune insertion détectée.")

else:

    for character, count in insertions.most_common(20):

        print(
            f"{repr(character)} : {count}"
        )


# ============================================================
# 11. ERREURS PAR LONGUEUR DE MOT
# ============================================================

print()
print("=" * 60)
print("ERREURS PAR LONGUEUR DE MOT WOLOF")
print("=" * 60)


length_counter = Counter()


for _, row in df.iterrows():

    wolof = str(
        row["Wolof"]
    )

    words = wolof.split()

    for word in words:

        length_counter[len(word)] += 1


for length, count in sorted(
    length_counter.items()
):

    print(
        f"Longueur {length} caractère(s) : {count}"
    )


# ============================================================
# 12. PHRASES AVEC LES PLUS GRANDES DIFFERENCES
# ============================================================

print()
print("=" * 60)
print("EXEMPLES AVEC LES PLUS GRANDES DIFFERENCES")
print("=" * 60)


difference_list = []


for _, row in df.iterrows():

    expected = str(
        row["Ajami_attendu"]
    )

    predicted = str(
        row["Ajami_predit"]
    )

    distance = levenshtein_operations(
        expected,
        predicted
    )

    difference_list.append(
        (
            len(distance),
            str(row["Wolof"]),
            expected,
            predicted
        )
    )


difference_list.sort(
    reverse=True
)


for index, item in enumerate(
    difference_list[:20],
    start=1
):

    distance, wolof, expected, predicted = item

    print()
    print("Cas", index)
    print("Distance :", distance)
    print("Wolof     :", wolof)
    print("Attendu   :", expected)
    print("Prédit    :", predicted)


# ============================================================
# 13. SAUVEGARDE DES CONFUSIONS
# ============================================================

print()
print("=" * 60)
print("SAUVEGARDE DES ANALYSES")
print("=" * 60)


confusion_rows = []


for (
    (expected_char, predicted_char),
    count
) in substitutions.most_common() : confusion_rows.append({
        "caractere_attendu": expected_char,
        "caractere_predit": predicted_char,
        "nombre": count
    })


confusion_df = pd.DataFrame(
    confusion_rows
)


confusion_file = (
    "saved_models/character_confusions.csv"
)


confusion_df.to_csv(
    confusion_file,
    index=False,
    encoding="utf-8-sig"
)


print(
    "Confusions sauvegardées dans :"
)

print(
    confusion_file
)


# ============================================================
# 14. FIN
# ============================================================

print()
print("=" * 60)
print("ANALYSE TERMINEE")
print("=" * 60)