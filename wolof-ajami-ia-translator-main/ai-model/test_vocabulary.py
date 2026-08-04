from utils.vocabulary import Vocabulary


# Chargement vocabulaire Wolof
vocab_wolof = Vocabulary(
    "data/vocab_wolof.json"
)

# Chargement vocabulaire Ajami
vocab_ajami = Vocabulary(
    "data/vocab_ajami.json"
)


print("Taille vocabulaire Wolof :", len(vocab_wolof))
print("Taille vocabulaire Ajami :", len(vocab_ajami))


# Test encodage
texte = "taxawal"

encoded = vocab_wolof.encode(texte)

print("\nTexte :", texte)
print("Indices :", encoded)


# Test décodage
decoded = vocab_wolof.decode(encoded)

print("Décodage :", decoded)