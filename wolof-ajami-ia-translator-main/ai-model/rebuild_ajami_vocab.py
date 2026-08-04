import json
import pandas as pd

from config.config import (
    CORPUS_FILE,
    AJAMI_VOCAB_FILE,
)


# ============================================================
# RECONSTRUCTION DU VOCABULAIRE AJAMI
# ============================================================

print("=" * 60)
print("RECONSTRUCTION DU VOCABULAIRE AJAMI")
print("=" * 60)


# ============================================================
# 1. CHARGEMENT DU CORPUS
# ============================================================

print()
print("Chargement du corpus...")


df = pd.read_csv(
    CORPUS_FILE,
    encoding="utf-8"
)


print(
    "Nombre de phrases :",
    len(df)
)


# ============================================================
# 2. RECUPERATION DES CARACTERES AJAMI
# ============================================================

characters = set()


for phrase in df["ajami"].astype(str):

    for caractere in phrase:

        characters.add(
            caractere
        )


# ============================================================
# 3. TOKENS SPECIAUX
# ============================================================

special_tokens = [
    "<PAD>",
    "<SOS>",
    "<EOS>",
    "<UNK>"
]


# ============================================================
# 4. CONSTRUCTION DU VOCABULAIRE
# ============================================================

# Les caractères sont triés pour obtenir
# un ordre stable et reproductible.

characters = sorted(
    characters
)


itos = (
    special_tokens
    + characters
)


stoi = {
    caractere: index
    for index, caractere in enumerate(itos)
}


vocab = {
    "stoi": stoi,
    "itos": itos
}


# ============================================================
# 5. SAUVEGARDE
# ============================================================

with open(
    AJAMI_VOCAB_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        vocab,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 6. AFFICHAGE
# ============================================================

print()
print("=" * 60)
print("NOUVEAU VOCABULAIRE AJAMI")
print("=" * 60)


print(
    "Nombre total de tokens :",
    len(itos)
)


print()
print("Index -> Caractère")


for index, caractere in enumerate(itos):

    print(
        index,
        "->",
        repr(caractere)
    )


# ============================================================
# 7. VERIFICATION DES CARACTERES SUPPRIMES
# ============================================================

print()
print("=" * 60)
print("VERIFICATION")
print("=" * 60)


for caractere in ["á", "ò"]:

    if caractere in stoi:

        print(
            repr(caractere),
            "-> ERREUR : encore présent"
        )

    else:

        print(
            repr(caractere),
            "-> OK : absent du vocabulaire Ajami"
        )


# ============================================================
# 8. TOKENS SPECIAUX
# ============================================================

print()
print("=" * 60)
print("VERIFICATION DES TOKENS SPECIAUX")
print("=" * 60)


for token in special_tokens:

    if token in stoi:

        print(
            token,
            "-> OK"
        )

    else:

        print(
            token,
            "-> ERREUR"
        )


# ============================================================
# 9. FIN
# ============================================================

print()
print("=" * 60)
print("RECONSTRUCTION TERMINEE")
print("=" * 60)