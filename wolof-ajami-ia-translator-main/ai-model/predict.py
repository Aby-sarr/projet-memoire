import json
import torch

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
# 1. CONFIGURATION
# ============================================================

SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 60)
print("PREDICTION - WOLOF LATIN -> WOLOF AJAMI")
print("=" * 60)

print("Device :", DEVICE)
print("Seed :", SEED)


# ============================================================
# 2. CHARGEMENT DES VOCABULAIRES
# ============================================================

print()
print("=" * 60)
print("CHARGEMENT DES VOCABULAIRES")
print("=" * 60)


with open(
    WOLOF_VOCAB_FILE,
    "r",
    encoding="utf-8"
) as f:

    wolof_vocab = json.load(f)


with open(
    AJAMI_VOCAB_FILE,
    "r",
    encoding="utf-8"
) as f:

    ajami_vocab = json.load(f)


wolof_stoi = wolof_vocab["stoi"]
wolof_itos = wolof_vocab["itos"]

ajami_stoi = ajami_vocab["stoi"]
ajami_itos = ajami_vocab["itos"]


# ------------------------------------------------------------
# Sécurité : conversion éventuelle de itos
# ------------------------------------------------------------

if isinstance(wolof_itos, dict):

    wolof_itos = {
        int(index): caractere
        for index, caractere in wolof_itos.items()
    }


if isinstance(ajami_itos, dict):

    ajami_itos = {
        int(index): caractere
        for index, caractere in ajami_itos.items()
    }


print(
    "Vocabulaire Wolof :",
    len(wolof_stoi)
)

print(
    "Vocabulaire Ajami :",
    len(ajami_stoi)
)


# ============================================================
# 3. VERIFICATION DES TOKENS SPECIAUX
# ============================================================

print()
print("=" * 60)
print("VERIFICATION DES TOKENS SPECIAUX")
print("=" * 60)


required_wolof_tokens = [
    "<PAD>",
    "<SOS>",
    "<EOS>",
    "<UNK>"
]


required_ajami_tokens = [
    "<PAD>",
    "<SOS>",
    "<EOS>"
]


for token in required_wolof_tokens:

    if token not in wolof_stoi:

        raise ValueError(
            f"Token Wolof manquant dans le vocabulaire : {token}"
        )


for token in required_ajami_tokens:

    if token not in ajami_stoi:

        raise ValueError(
            f"Token Ajami manquant dans le vocabulaire : {token}"
        )


print("Tokens spéciaux Wolof : OK")
print("Tokens spéciaux Ajami : OK")


# ============================================================
# 4. CONSTRUCTION DU MODELE
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
    output_dim=len(ajami_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)


model = Seq2Seq(
    encoder,
    decoder,
    attention,
    DEVICE
).to(DEVICE)


print(
    "Modèle créé avec succès !"
)


# ============================================================
# 5. CHARGEMENT DU MEILLEUR MODELE
# ============================================================

MODEL_PATH = (
    MODEL_DIR / "best_model.pt"
)


print()
print("=" * 60)
print("CHARGEMENT DU MEILLEUR MODELE")
print("=" * 60)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Modèle introuvable : {MODEL_PATH}"
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print(
    "Modèle chargé avec succès !"
)


if "epoch" in checkpoint:

    print(
        "Epoch du meilleur modèle :",
        checkpoint["epoch"]
    )


# Le nom peut être valid_loss ou val_loss
if "valid_loss" in checkpoint:

    print(
        "Loss validation :",
        checkpoint["valid_loss"]
    )

elif "val_loss" in checkpoint:

    print(
        "Loss validation :",
        checkpoint["val_loss"]
    )


# ============================================================
# 6. FONCTION DE PREDICTION
# ============================================================

def predict(
    sentence,
    max_length=100
):
    """
    Traduit une phrase Wolof Latin
    vers le Wolof Ajami.
    """

    sentence = str(sentence)


    # --------------------------------------------------------
    # Encodage du texte source
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


    source_indices = (
        [wolof_stoi["<SOS>"]]
        + tokens
        + [wolof_stoi["<EOS>"]]
    )


    source = torch.tensor(
        source_indices,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)


    # --------------------------------------------------------
    # Encodage
    # --------------------------------------------------------

    with torch.no_grad():

        encoder_outputs, hidden = (
            model.encoder(source)
        )


    # --------------------------------------------------------
    # Premier token du décodeur
    # --------------------------------------------------------

    input_token = torch.tensor(
        [ajami_stoi["<SOS>"]],
        dtype=torch.long,
        device=DEVICE
    )


    resultat = []


    # --------------------------------------------------------
    # Décodage caractère par caractère
    # --------------------------------------------------------

    for step in range(max_length):

        with torch.no_grad():

            context, attention_weights = (
                model.attention(
                    hidden,
                    encoder_outputs
                )
            )


            output, hidden = (
                model.decoder(
                    input_token,
                    hidden,
                    context
                )
            )


        predicted_token = (
            output.argmax(
                dim=1
            ).item()
        )


        # Sécurité contre un indice invalide

        if (
            predicted_token < 0
            or predicted_token >= len(ajami_itos)
        ):

            break


        predicted_char = ajami_itos[
            predicted_token
        ]


        # ----------------------------------------------------
        # Fin de séquence
        # ----------------------------------------------------

        if predicted_char == "<EOS>":

            break


        # ----------------------------------------------------
        # Ignorer les tokens spéciaux
        # ----------------------------------------------------

        if predicted_char not in [
            "<PAD>",
            "<SOS>"
        ]:

            resultat.append(
                predicted_char
            )


        # ----------------------------------------------------
        # Le token prédit devient
        # l'entrée suivante du décodeur
        # ----------------------------------------------------

        input_token = torch.tensor(
            [predicted_token],
            dtype=torch.long,
            device=DEVICE
        )


    return "".join(
        resultat
    )


# ============================================================
# 7. TESTS AUTOMATIQUES
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


for i, phrase in enumerate(
    tests,
    start=1
):

    resultat = predict(
        phrase,
        max_length=100
    )


    print()
    print(
        "Test",
        i
    )

    print(
        "Wolof :",
        phrase
    )

    print(
        "Ajami :",
        resultat
    )


# ============================================================
# 8. MODE INTERACTIF
# ============================================================

print()
print("=" * 60)
print("MODE INTERACTIF")
print("=" * 60)


print(
    "Écris un mot ou une phrase en Wolof."
)

print(
    "Tape 'quit' pour arrêter le programme."
)

print()


while True:

    phrase = input(
        "Wolof : "
    ).strip()


    if phrase.lower() == "quit":

        print()
        print(
            "Programme terminé."
        )

        break


    if phrase == "":

        print(
            "Veuillez entrer un mot ou une phrase."
        )

        continue


    resultat = predict(
        phrase,
        max_length=100
    )


    print(
        "Ajami :",
        resultat
    )

    print()