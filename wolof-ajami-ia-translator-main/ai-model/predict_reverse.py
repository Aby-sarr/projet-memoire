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
print("PREDICTION REVERSE - WOLOF AJAMI -> WOLOF LATIN")
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


# SOURCE = AJAMI
ajami_stoi = ajami_vocab["stoi"]
ajami_itos = ajami_vocab["itos"]


# CIBLE = WOLOF LATIN
wolof_stoi = wolof_vocab["stoi"]
wolof_itos = wolof_vocab["itos"]


# ============================================================
# 3. NORMALISATION DES itos
# ============================================================

if isinstance(ajami_itos, dict):

    ajami_itos = {
        int(index): caractere
        for index, caractere in ajami_itos.items()
    }


if isinstance(wolof_itos, dict):

    wolof_itos = {
        int(index): caractere
        for index, caractere in wolof_itos.items()
    }


print(
    "Vocabulaire Ajami :",
    len(ajami_stoi)
)

print(
    "Vocabulaire Wolof :",
    len(wolof_stoi)
)


# ============================================================
# 4. TOKENS SPECIAUX
# ============================================================

print()
print("=" * 60)
print("VERIFICATION DES TOKENS SPECIAUX")
print("=" * 60)


required_ajami_tokens = [
    "<PAD>",
    "<SOS>",
    "<EOS>",
    "<UNK>",
]


required_wolof_tokens = [
    "<PAD>",
    "<SOS>",
    "<EOS>",
    "<UNK>",
]


for token in required_ajami_tokens:

    if token not in ajami_stoi:

        raise ValueError(
            f"Token Ajami manquant : {token}"
        )


for token in required_wolof_tokens:

    if token not in wolof_stoi:

        raise ValueError(
            f"Token Wolof manquant : {token}"
        )


print("Tokens spéciaux Ajami : OK")
print("Tokens spéciaux Wolof : OK")


# ============================================================
# 5. AFFICHAGE DES INDICES IMPORTANTS
# ============================================================

print()
print("=" * 60)
print("INDICES DES TOKENS WOLOF")
print("=" * 60)


print(
    "<PAD> =",
    wolof_stoi["<PAD>"]
)

print(
    "<SOS> =",
    wolof_stoi["<SOS>"]
)

print(
    "<EOS> =",
    wolof_stoi["<EOS>"]
)

print(
    "<UNK> =",
    wolof_stoi["<UNK>"]
)


# Vérification du caractère C
if "C" in wolof_stoi:

    print(
        "C =",
        wolof_stoi["C"]
    )

    print(
        "itos[C] =",
        wolof_itos[wolof_stoi["C"]]
    )


# ============================================================
# 6. CONSTRUCTION DU MODELE REVERSE
# ============================================================

print()
print("=" * 60)
print("CONSTRUCTION DU MODELE REVERSE")
print("=" * 60)


# ------------------------------------------------------------
# ENCODEUR
# SOURCE = AJAMI
# ------------------------------------------------------------

encoder = EncoderGRU(
    input_dim=len(ajami_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)


# ------------------------------------------------------------
# ATTENTION
# ------------------------------------------------------------

attention = BahdanauAttention(
    HIDDEN_SIZE
)


# ------------------------------------------------------------
# DECODEUR
# CIBLE = WOLOF LATIN
# ------------------------------------------------------------

decoder = DecoderGRU(
    output_dim=len(wolof_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)


# ------------------------------------------------------------
# SEQ2SEQ
# ------------------------------------------------------------

model = Seq2Seq(
    encoder,
    decoder,
    attention,
    DEVICE
).to(DEVICE)


print(
    "Modèle reverse créé avec succès !"
)


# ============================================================
# 7. CHARGEMENT DE BEST_MODEL_REVERSE
# ============================================================

MODEL_PATH = (
    MODEL_DIR / "best_model_reverse.pt"
)


print()
print("=" * 60)
print("CHARGEMENT DE BEST_MODEL_REVERSE")
print("=" * 60)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Modèle reverse introuvable : {MODEL_PATH}"
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ------------------------------------------------------------
# Le checkpoint d'entraînement contient :
# model_state_dict
# ------------------------------------------------------------

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Modèle reverse chargé avec succès !"
    )

    if "epoch" in checkpoint:

        print(
            "Meilleure epoch :",
            checkpoint["epoch"]
        )

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

else:

    model.load_state_dict(
        checkpoint
    )

    print(
        "Checkpoint chargé directement."
    )


model.eval()


# ============================================================
# 8. ENCODAGE AJAMI
# ============================================================

def encode_ajami(text):
    """
    Convertit le texte Ajami en indices.

    Ajoute :
        <SOS> au début
        <EOS> à la fin
    """

    text = str(text)

    tokens = []

    for caractere in text:

        if caractere in ajami_stoi:

            tokens.append(
                ajami_stoi[caractere]
            )

        else:

            tokens.append(
                ajami_stoi["<UNK>"]
            )


    source_indices = (
        [ajami_stoi["<SOS>"]]
        + tokens
        + [ajami_stoi["<EOS>"]]
    )


    source = torch.tensor(
        source_indices,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)


    return source


# ============================================================
# 9. DECODAGE WOLOF
# ============================================================

def decode_wolof(indices):
    """
    Convertit les indices prédits en Wolof Latin.
    """

    resultat = []

    for indice in indices:

        indice = int(indice)

        if isinstance(wolof_itos, list):

            if (
                indice < 0
                or indice >= len(wolof_itos)
            ):
                continue

            caractere = wolof_itos[indice]

        else:

            caractere = wolof_itos.get(
                indice,
                "<UNK>"
            )


        if caractere == "<EOS>":

            break


        if caractere in [
            "<PAD>",
            "<SOS>",
            "<UNK>",
        ]:

            continue


        resultat.append(
            caractere
        )


    return "".join(resultat)


# ============================================================
# 10. PREDICTION REVERSE
# ============================================================

def predict(
    sentence,
    max_length=100
):
    """
    Translittération :

        Wolof Ajami -> Wolof Latin
    """

    sentence = str(sentence)

    # --------------------------------------------------------
    # Encodage source
    # --------------------------------------------------------

    source = encode_ajami(
        sentence
    )


    # --------------------------------------------------------
    # Encodeur
    # --------------------------------------------------------

    with torch.no_grad():

        encoder_outputs, hidden = (
            model.encoder(source)
        )


    # --------------------------------------------------------
    # Premier token du décodeur
    # --------------------------------------------------------

    input_token = torch.tensor(
        [wolof_stoi["<SOS>"]],
        dtype=torch.long,
        device=DEVICE
    )


    predicted_indices = []


    # --------------------------------------------------------
    # Décodage auto-régressif
    # --------------------------------------------------------

    for step in range(max_length):

        with torch.no_grad():

            # Attention Bahdanau

            context, attention_weights = (
                model.attention(
                    hidden,
                    encoder_outputs
                )
            )


            # Décodeur GRU

            output, hidden = (
                model.decoder(
                    input_token,
                    hidden,
                    context
                )
            )


        # ----------------------------------------------------
        # Token ayant la probabilité maximale
        # ----------------------------------------------------

        predicted_token = (
            output.argmax(
                dim=1
            ).item()
        )


        # ----------------------------------------------------
        # Sécurité indice
        # ----------------------------------------------------

        if (
            predicted_token < 0
            or predicted_token >= len(wolof_itos)
        ):

            break


        predicted_char = (
            wolof_itos[predicted_token]
        )


        # ----------------------------------------------------
        # EOS
        # ----------------------------------------------------

        if predicted_char == "<EOS>":

            break


        # ----------------------------------------------------
        # Tokens spéciaux
        # ----------------------------------------------------

        if predicted_char not in [
            "<PAD>",
            "<SOS>",
            "<UNK>",
        ]:

            predicted_indices.append(
                predicted_token
            )


        # ----------------------------------------------------
        # La prédiction devient
        # l'entrée suivante
        # ----------------------------------------------------

        input_token = torch.tensor(
            [predicted_token],
            dtype=torch.long,
            device=DEVICE
        )


    # --------------------------------------------------------
    # Conversion indices -> texte
    # --------------------------------------------------------

    return decode_wolof(
        predicted_indices
    )


# ============================================================
# 11. DIAGNOSTIC DETAILLE
# ============================================================

def diagnostic_prediction(sentence):
    """
    Affiche les indices et caractères prédits.
    Utile pour comprendre pourquoi le modèle
    produit une sortie répétitive.
    """

    print()
    print("-" * 60)
    print("DIAGNOSTIC")
    print("-" * 60)

    print(
        "Entrée Ajami :",
        sentence
    )


    source = encode_ajami(
        sentence
    )


    print(
        "Indices source :",
        source.squeeze(0).tolist()
    )


    predicted_indices = []


    with torch.no_grad():

        encoder_outputs, hidden = (
            model.encoder(source)
        )


        input_token = torch.tensor(
            [wolof_stoi["<SOS>"]],
            dtype=torch.long,
            device=DEVICE
        )


        for step in range(
            min(30, max(1, len(sentence) * 3))
        ):

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


            if (
                predicted_token < 0
                or predicted_token >= len(wolof_itos)
            ):

                break


            predicted_char = (
                wolof_itos[predicted_token]
            )


            print(
                f"Step {step:02d} | "
                f"Indice = {predicted_token:02d} | "
                f"Caractère = {repr(predicted_char)}"
            )


            if predicted_char == "<EOS>":

                break


            if predicted_char not in [
                "<PAD>",
                "<SOS>",
                "<UNK>",
            ]:

                predicted_indices.append(
                    predicted_token
                )


            input_token = torch.tensor(
                [predicted_token],
                dtype=torch.long,
                device=DEVICE
            )


    resultat = decode_wolof(
        predicted_indices
    )


    print()
    print(
        "Indices prédits :",
        predicted_indices
    )


    print(
        "Résultat Wolof :",
        resultat
    )


# ============================================================
# 12. TESTS AUTOMATIQUES
# ============================================================

print()
print("=" * 60)
print("TESTS AUTOMATIQUES REVERSE")
print("=" * 60)


tests = [
    "گوددي",
    "دێپپ",
    "يوخوي",
    "چێرام",
    "باريوول",
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
        f"Test {i}/{len(tests)}"
    )


    print(
        "Ajami :",
        phrase
    )


    print(
        "Wolof :",
        resultat
    )


# ============================================================
# 13. DIAGNOSTIC DU PREMIER TEST
# ============================================================

print()
print("=" * 60)
print("DIAGNOSTIC DU PREMIER TEST")
print("=" * 60)


diagnostic_prediction(
    "گوددي"
)


# ============================================================
# 14. MODE INTERACTIF
# ============================================================

print()
print("=" * 60)
print("MODE INTERACTIF REVERSE")
print("=" * 60)


print(
    "Écris un mot ou une phrase en Wolof Ajami."
)

print(
    "Tape 'quit' pour arrêter le programme."
)

print()


while True:

    phrase = input(
        "Ajami : "
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
        "Wolof :",
        resultat
    )

    print()