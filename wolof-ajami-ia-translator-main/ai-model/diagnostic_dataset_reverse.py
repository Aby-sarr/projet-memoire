import pandas as pd
import json
from collections import Counter

from config.config import (
    CORPUS_FILE,
    AJAMI_VOCAB_FILE,
    WOLOF_VOCAB_FILE,
)


print("=" * 70)
print("DIAGNOSTIC DATASET REVERSE")
print("AJAMI -> WOLOF LATIN")
print("=" * 70)


# ============================================================
# 1. CHARGEMENT
# ============================================================

df = pd.read_csv(
    CORPUS_FILE,
    encoding="utf-8"
)

print()
print("Colonnes du corpus :")
print(df.columns.tolist())

print()
print("Nombre de lignes :", len(df))


# ============================================================
# 2. VERIFICATION DES COLONNES
# ============================================================

if "ajami" not in df.columns:
    raise ValueError("Colonne 'ajami' introuvable.")

if "Wolof" not in df.columns:
    raise ValueError("Colonne 'Wolof' introuvable.")


# ============================================================
# 3. AFFICHER QUELQUES PAIRES
# ============================================================

print()
print("=" * 70)
print("PREMIERES PAIRES DU CORPUS")
print("=" * 70)

for i in range(min(20, len(df))):

    ajami = str(df.iloc[i]["ajami"])
    wolof = str(df.iloc[i]["Wolof"])

    print()
    print(f"{i + 1}.")
    print("Ajami :", repr(ajami))
    print("Wolof :", repr(wolof))

    if len(ajami) > 0:
        print("Premier caractère Ajami :", repr(ajami[0]))

    if len(wolof) > 0:
        print("Premier caractère Wolof :", repr(wolof[0]))


# ============================================================
# 4. FREQUENCE PREMIER CARACTERE WOLOF
# ============================================================

print()
print("=" * 70)
print("FREQUENCE DU PREMIER CARACTERE WOLOF")
print("=" * 70)

first_wolof_chars = []

for text in df["Wolof"].astype(str):

    if len(text) > 0:
        first_wolof_chars.append(text[0])


counter_wolof = Counter(first_wolof_chars)


print()

for char, count in counter_wolof.most_common(30):

    percentage = (
        count / len(first_wolof_chars)
    ) * 100

    print(
        repr(char),
        "->",
        count,
        f"({percentage:.2f} %)"
    )


# ============================================================
# 5. FREQUENCE PREMIER CARACTERE AJAMI
# ============================================================

print()
print("=" * 70)
print("FREQUENCE DU PREMIER CARACTERE AJAMI")
print("=" * 70)

first_ajami_chars = []

for text in df["ajami"].astype(str):

    if len(text) > 0:
        first_ajami_chars.append(text[0])


counter_ajami = Counter(first_ajami_chars)


print()

for char, count in counter_ajami.most_common(30):

    percentage = (
        count / len(first_ajami_chars)
    ) * 100

    print(
        repr(char),
        "->",
        count,
        f"({percentage:.2f} %)"
    )


# ============================================================
# 6. VERIFICATION DE C
# ============================================================

print()
print("=" * 70)
print("VERIFICATION DU CARACTERE C")
print("=" * 70)


c_count = 0
lower_c_count = 0


for text in df["Wolof"].astype(str):

    c_count += text.count("C")
    lower_c_count += text.count("c")


print("Nombre de 'C' :", c_count)
print("Nombre de 'c' :", lower_c_count)


# ============================================================
# 7. LIGNES COMMENCANT PAR C
# ============================================================

print()
print("=" * 70)
print("PHRASES WOLOF COMMENCANT PAR C")
print("=" * 70)


starts_c = df[
    df["Wolof"]
    .astype(str)
    .str.startswith("C")
]


print(
    "Nombre de phrases commençant par C :",
    len(starts_c)
)


print(
    "Pourcentage :",
    f"{len(starts_c) / len(df) * 100:.2f} %"
)


print()

for i in range(min(20, len(starts_c))):

    row = starts_c.iloc[i]

    print(
        "Ajami :",
        repr(str(row["ajami"]))
    )

    print(
        "Wolof :",
        repr(str(row["Wolof"]))
    )

    print()


# ============================================================
# 8. VOCABULAIRES
# ============================================================

print()
print("=" * 70)
print("VERIFICATION VOCABULAIRES")
print("=" * 70)


with open(
    AJAMI_VOCAB_FILE,
    "r",
    encoding="utf-8"
) as f:

    ajami_vocab = json.load(f)


with open(
    WOLOF_VOCAB_FILE,
    "r",
    encoding="utf-8"
) as f:

    wolof_vocab = json.load(f)


print()
print("Ajami vocab :", len(ajami_vocab["stoi"]))
print("Wolof vocab :", len(wolof_vocab["stoi"]))


print()
print("Wolof :")
print("C =", wolof_vocab["stoi"].get("C"))
print("c =", wolof_vocab["stoi"].get("c"))
print("a =", wolof_vocab["stoi"].get("a"))
print("n =", wolof_vocab["stoi"].get("n"))
print("SOS =", wolof_vocab["stoi"].get("<SOS>"))
print("EOS =", wolof_vocab["stoi"].get("<EOS>"))
print("UNK =", wolof_vocab["stoi"].get("<UNK>"))


# ============================================================
# FIN
# ============================================================

print()
print("=" * 70)
print("FIN DU DIAGNOSTIC")
print("=" * 70)