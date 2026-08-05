import pandas as pd
from collections import Counter
from pathlib import Path
import re

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("saved_models/audit_errors_reverse.csv")

OUTPUT_DETAIL = Path("saved_models/linguistic_errors_detail_reverse.csv")
OUTPUT_SUMMARY = Path("saved_models/linguistic_errors_summary_reverse.csv")

# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 60)
print("ANALYSE DETAILLEE DES ERREURS LINGUISTIQUES")
print("AJAMI -> WOLOF LATIN")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print()
print("Nombre total de lignes :", len(df))

print()
print("Colonnes détectées :")
for col in df.columns:
    print(" -", col)

# ============================================================
# DETECTION DES COLONNES
# ============================================================

source_col = "Ajami"
expected_col = "Wolof_attendu"
predicted_col = "Wolof_predit"

for col in [source_col, expected_col, predicted_col]:
    if col not in df.columns:
        raise ValueError(f"Colonne introuvable : {col}")

# ============================================================
# GARDER UNIQUEMENT LES ERREURS LINGUISTIQUES
# ============================================================

if "categorie_audit" in df.columns:
    linguistic_categories = [
        "Erreur linguistique / structurelle"
    ]

    linguistic_df = df[
        df["categorie_audit"].isin(linguistic_categories)
    ].copy()

elif "audit_category" in df.columns:
    linguistic_df = df[
        df["audit_category"] == "Erreur linguistique / structurelle"
    ].copy()

else:
    raise ValueError(
        "Colonne de catégorie d'audit introuvable. "
        "Vérifie le nom de la colonne dans audit_errors_reverse.csv."
    )

print()
print("=" * 60)
print("ERREURS LINGUISTIQUES / STRUCTURELLES")
print("=" * 60)

print()
print("Nombre d'erreurs analysées :", len(linguistic_df))

# ============================================================
# FONCTION : DISTANCE DE LEVENSHTEIN
# ============================================================

def levenshtein_ops(expected, predicted):
    """
    Retourne le nombre de :
    - substitutions
    - insertions
    - suppressions
    - égalités
    """

    a = list(str(expected))
    b = list(str(predicted))

    n = len(a)
    m = len(b)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    # Reconstruction
    i = n
    j = m

    substitutions = 0
    insertions = 0
    deletions = 0
    matches = 0

    while i > 0 or j > 0:

        if i > 0 and j > 0:

            cost = 0 if a[i - 1] == b[j - 1] else 1

            if dp[i][j] == dp[i - 1][j - 1] + cost:

                if cost == 0:
                    matches += 1
                else:
                    substitutions += 1

                i -= 1
                j -= 1
                continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
            continue

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
            continue

        break

    return substitutions, insertions, deletions, matches


# ============================================================
# ANALYSE DES ERREURS
# ============================================================

results = []

category_counter = Counter()
substitution_counter = Counter()
word_counter = Counter()

for _, row in linguistic_df.iterrows():

    expected = str(row[expected_col])
    predicted = str(row[predicted_col])

    substitutions, insertions, deletions, matches = \
        levenshtein_ops(expected, predicted)

    # --------------------------------------------------------
    # Catégorie principale
    # --------------------------------------------------------

    if substitutions > 0 and insertions == 0 and deletions == 0:
        category = "Substitution"

    elif insertions > 0 and substitutions == 0 and deletions == 0:
        category = "Insertion"

    elif deletions > 0 and substitutions == 0 and insertions == 0:
        category = "Suppression"

    elif (
        substitutions == 0
        and insertions > 0
        and deletions > 0
    ):
        category = "Insertion + suppression"

    elif (
        substitutions > 0
        and insertions > 0
        and deletions == 0
    ):
        category = "Substitution + insertion"

    elif (
        substitutions > 0
        and deletions > 0
        and insertions == 0
    ):
        category = "Substitution + suppression"

    else:
        category = "Erreur multiple"

    category_counter[category] += 1

    # --------------------------------------------------------
    # Mots concernés
    # --------------------------------------------------------

    expected_words = re.findall(
        r"\S+",
        expected
    )

    predicted_words = re.findall(
        r"\S+",
        predicted
    )

    for word in expected_words:
        word_counter[word] += 1

    # --------------------------------------------------------
    # Substitutions caractère-à-caractère
    # --------------------------------------------------------

    a = list(expected)
    b = list(predicted)

    n = len(a)
    m = len(b)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    i = n
    j = m

    local_substitutions = []

    while i > 0 and j > 0:

        cost = 0 if a[i - 1] == b[j - 1] else 1

        if dp[i][j] == dp[i - 1][j - 1] + cost:

            if cost == 1:
                substitution_counter[
                    (a[i - 1], b[j - 1])
                ] += 1

                local_substitutions.append(
                    f"{a[i - 1]}->{b[j - 1]}"
                )

            i -= 1
            j -= 1

        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1

        else:
            j -= 1

    results.append({
        source_col: row[source_col],
        expected_col: expected,
        predicted_col: predicted,
        "categorie_detaillee": category,
        "substitutions": substitutions,
        "insertions": insertions,
        "suppressions": deletions,
        "correspondances": matches,
        "confusions": ", ".join(local_substitutions)
    })


# ============================================================
# RESULTATS
# ============================================================

result_df = pd.DataFrame(results)

print()
print("=" * 60)
print("CATEGORIES D'ERREURS")
print("=" * 60)

total = len(result_df)

for category, count in category_counter.most_common():

    percentage = (
        count / total * 100
        if total > 0 else 0
    )

    print(
        f"{category:<35} "
        f"{count:>4} "
        f"({percentage:>6.2f} %)"
    )

print()
print("-" * 60)
print("TOTAL :", total)

# ============================================================
# CONFUSIONS DE CARACTERES
# ============================================================

print()
print("=" * 60)
print("PRINCIPALES CONFUSIONS DE CARACTERES")
print("=" * 60)

for (expected_char, predicted_char), count in \
        substitution_counter.most_common(30):

    print(
        f"'{expected_char}' -> '{predicted_char}'"
        f" : {count}"
    )

# ============================================================
# MOTS LES PLUS CONCERNES
# ============================================================

print()
print("=" * 60)
print("MOTS LES PLUS FREQUEMMENT CONCERNES")
print("=" * 60)

for word, count in word_counter.most_common(30):

    print(
        f"{word:<30} : {count}"
    )

# ============================================================
# SAUVEGARDE DETAIL
# ============================================================

result_df.to_csv(
    OUTPUT_DETAIL,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# SAUVEGARDE RESUME
# ============================================================

summary_rows = []

for category, count in category_counter.most_common():

    percentage = (
        count / total * 100
        if total > 0 else 0
    )

    summary_rows.append({
        "categorie": category,
        "nombre": count,
        "pourcentage": round(percentage, 2)
    })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# FIN
# ============================================================

print()
print("=" * 60)
print("FICHIERS SAUVEGARDES")
print("=" * 60)

print()
print("Détail :", OUTPUT_DETAIL)
print("Résumé :", OUTPUT_SUMMARY)

print()
print("ANALYSE TERMINEE")