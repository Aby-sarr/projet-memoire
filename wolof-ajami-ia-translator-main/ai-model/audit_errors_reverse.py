import pandas as pd
import unicodedata
import re

INPUT = "saved_models/test_errors_reverse.csv"
OUTPUT_DETAIL = "saved_models/audit_errors_reverse.csv"
OUTPUT_SUMMARY = "saved_models/audit_summary_reverse.csv"


# ============================================================
# NORMALISATION
# ============================================================

def normalize_case(text):
    return str(text).lower()


def normalize_accents(text):
    text = str(text)
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def normalize_case_accents(text):
    return normalize_accents(normalize_case(text))


def normalize_spaces(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_punctuation(text):
    text = str(text)
    return re.sub(r'\s+', ' ', text).strip()


# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 60)
print("AUDIT DES ERREURS - AJAMI -> WOLOF LATIN")
print("=" * 60)

df = pd.read_csv(INPUT)

expected = "Wolof_attendu"
predicted = "Wolof_predit"

print()
print("Nombre total d'erreurs :", len(df))


# ============================================================
# CLASSIFICATION
# ============================================================

results = []

for _, row in df.iterrows():

    attendu = str(row[expected])
    predit = str(row[predicted])

    exact = attendu == predit

    case_only = (
        not exact
        and normalize_case(attendu) == normalize_case(predit)
    )

    accent_only = (
        not exact
        and not case_only
        and normalize_accents(attendu) == normalize_accents(predit)
    )

    case_accent_only = (
        not exact
        and normalize_case_accents(attendu)
        == normalize_case_accents(predit)
    )

    spaces_only = (
        not exact
        and normalize_spaces(attendu)
        == normalize_spaces(predit)
    )

    punctuation_only = (
        not exact
        and re.sub(r"[^\w\s]", "", attendu)
        == re.sub(r"[^\w\s]", "", predit)
    )

    # Classification finale
    if case_only:
        category = "Casse uniquement"

    elif accent_only:
        category = "Diacritique uniquement"

    elif case_accent_only:
        category = "Casse + diacritique"

    elif spaces_only:
        category = "Espaces uniquement"

    elif punctuation_only:
        category = "Ponctuation uniquement"

    else:
        category = "Erreur linguistique / structurelle"

    results.append({
        "Ajami": row["Ajami"],
        "Wolof_attendu": attendu,
        "Wolof_predit": predit,
        "categorie_audit": category
    })


audit = pd.DataFrame(results)

audit.to_csv(
    OUTPUT_DETAIL,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# STATISTIQUES
# ============================================================

summary = (
    audit["categorie_audit"]
    .value_counts()
    .rename_axis("categorie")
    .reset_index(name="nombre")
)

summary["proportion"] = (
    summary["nombre"] / len(audit) * 100
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# AFFICHAGE
# ============================================================

print()
print("=" * 60)
print("RESULTATS DE L'AUDIT")
print("=" * 60)

for _, row in summary.iterrows():
    print(
        f"{row['categorie']:<35}"
        f"{int(row['nombre']):>6} "
        f"({row['proportion']:>6.2f} %)"
    )

print()
print("-" * 60)
print("TOTAL :", len(audit))

linguistic = (
    audit["categorie_audit"]
    == "Erreur linguistique / structurelle"
).sum()

print("Erreurs réellement linguistiques / structurelles :",
      linguistic)

print(
    "Proportion :",
    f"{linguistic / len(audit) * 100:.2f} %"
)

print()
print("=" * 60)
print("FICHIERS SAUVEGARDES")
print("=" * 60)

print("Détail :", OUTPUT_DETAIL)
print("Résumé :", OUTPUT_SUMMARY)

print()
print("AUDIT TERMINE")