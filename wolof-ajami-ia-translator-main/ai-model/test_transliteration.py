import torch
import json

from config.config import (
    WOLOF_VOCAB_FILE,
    AJAMI_VOCAB_FILE,
    MODEL_DIR,
    EMBEDDING,
    HIDDEN_SIZE,
    DEVICE,
)

from models.encoder import EncoderGRU
from models.attention import BahdanauAttention
from models.decoder import DecoderGRU
from models.seq2seq import Seq2Seq


# ============================================================
# 1. CHARGEMENT DES VOCABULAIRES
# ============================================================

print("=" * 60)
print("CHARGEMENT DES VOCABULAIRES")
print("=" * 60)

with open(WOLOF_VOCAB_FILE, "r", encoding="utf-8") as f:
    wolof_vocab = json.load(f)

with open(AJAMI_VOCAB_FILE, "r", encoding="utf-8") as f:
    ajami_vocab = json.load(f)

wolof_stoi = wolof_vocab["stoi"]
ajami_itos = ajami_vocab["itos"]

# Certains fichiers JSON peuvent contenir une liste
# et d'autres un dictionnaire.
if isinstance(ajami_itos, dict):
    ajami_itos = {
        int(index): caractere
        for index, caractere in ajami_itos.items()
    }

print("Vocabulaire Wolof :", len(wolof_stoi))
print("Vocabulaire Ajami :", len(ajami_itos))


# ============================================================
# 2. CONSTRUCTION DU MODELE
# ============================================================

print()
print("=" * 60)
print("CONSTRUCTION DU MODELE")
print("=" * 60)

encoder = EncoderGRU(
    input_dim=len(wolof_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)

attention = BahdanauAttention(
    HIDDEN_SIZE
)

decoder = DecoderGRU(
    output_dim=len(ajami_itos),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)

model = Seq2Seq(
    encoder,
    decoder,
    attention,
    DEVICE
).to(DEVICE)

print("Modèle créé avec succès !")


# ============================================================
# 3. CHARGEMENT DU MEILLEUR MODELE
# ============================================================

MODEL_PATH = MODEL_DIR / "best_model.pt"

print()
print("=" * 60)
print("CHARGEMENT DU MEILLEUR MODELE")
print("=" * 60)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Modèle chargé avec succès !")

if "epoch" in checkpoint:
    print("Epoch du meilleur modèle :", checkpoint["epoch"])

if "valid_loss" in checkpoint:
    print(
        "Loss validation :",
        checkpoint["valid_loss"]
    )


# ============================================================
# 4. FONCTION DE TRANSLITTERATION
# ============================================================

def transliterate(sentence):
    """
    Transforme une phrase Wolof Latin en Wolof Ajami.
    """

    sentence = sentence.strip()

    if not sentence:
        return ""

    # --------------------------------------------------------
    # Conversion des caractères Wolof en indices
    # --------------------------------------------------------

    tokens = []

    for caractere in sentence:

        if caractere in wolof_stoi:
            tokens.append(
                wolof_stoi[caractere]
            )

        else:
            tokens.append(
                wolof_stoi["<UNK>"]
            )

    # --------------------------------------------------------
    # Ajout de SOS et EOS
    # --------------------------------------------------------

    source_indices = [
        wolof_stoi["<SOS>"]
    ]

    source_indices.extend(tokens)

    source_indices.append(
        wolof_stoi["<EOS>"]
    )

    source = torch.tensor(
        source_indices,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    # --------------------------------------------------------
    # Création de la séquence cible
    # --------------------------------------------------------

    pad_idx = ajami_itos.index("<PAD>")
    sos_idx = ajami_itos.index("<SOS>")

    MAX_LENGTH = 100

    target = torch.full(
        (1, MAX_LENGTH),
        pad_idx,
        dtype=torch.long,
        device=DEVICE
    )

    target[0, 0] = sos_idx

    # --------------------------------------------------------
    # Prédiction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            source,
            target,
            teacher_forcing_ratio=0
        )

    # --------------------------------------------------------
    # Sélection du caractère ayant la probabilité maximale
    # --------------------------------------------------------

    predictions = output.argmax(
        dim=2
    )

    resultat = []

    for indice in predictions[0]:

        indice = indice.item()

        if indice < 0 or indice >= len(ajami_itos):
            continue

        caractere = ajami_itos[indice]

        # Fin de la prédiction
        if caractere == "<EOS>":
            break

        # On ignore les tokens spéciaux
        if caractere in [
            "<PAD>",
            "<SOS>"
        ]:
            continue

        resultat.append(caractere)

    return "".join(resultat)


# ============================================================
# 5. TESTS AUTOMATIQUES
# ============================================================

print()
print("=" * 60)
print("TESTS AUTOMATIQUES")
print("=" * 60)

tests = [
    "ndank",
    "yomb",
    "bopp",
    "kër",
    "ba",
    "am",
    "a",
    "yàlla",
    "nangu",
    "ñu",
    "ekool",
    "laay",
    "amee",
    "jërejëf",
    "dangeen",
    "réer",
    "yam",
    "taxufeex",
    "àtte",
    "aymusiba",
]

for numero, mot in enumerate(tests, start=1):

    prediction = transliterate(mot)

    print()
    print(f"Test {numero}")
    print("Wolof  :", mot)
    print("Ajami  :", prediction)


# ============================================================
# 6. MODE INTERACTIF
# ============================================================

print()
print("=" * 60)
print("MODE INTERACTIF")
print("=" * 60)

print("Écris un mot ou une phrase en Wolof.")
print("Tape 'quit' pour arrêter le programme.")
print()

while True:

    phrase = input("Wolof : ").strip()

    if phrase.lower() == "quit":
        break

    if not phrase:
        print("Veuillez saisir un mot ou une phrase.")
        continue

    resultat = transliterate(
        phrase
    )

    print("Ajami :", resultat)
    print()

print("=" * 60)
print("TEST TERMINÉ")
print("=" * 60)