import json
import torch

from config.config import (
    WOLOF_VOCAB_FILE,
    AJAMI_VOCAB_FILE,
    MODEL_DIR,
)

from models.encoder import EncoderGRU
from models.decoder import DecoderGRU
from models.attention import BahdanauAttention
from models.seq2seq import Seq2Seq


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EMBEDDING_DIM = 256
HIDDEN_DIM = 512

MODEL_PATH = MODEL_DIR / "best_model_reverse.pt"


# ============================================================
# CHARGEMENT VOCABULAIRE
# ============================================================

def load_vocab(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stoi = data["stoi"]
    itos = data["itos"]

    # Certains vocabulaires ont itos sous forme de dictionnaire
    if isinstance(itos, dict):
        itos = {
            int(k): v
            for k, v in itos.items()
        }

    return stoi, itos


# ============================================================
# CHARGEMENT VOCABULAIRES
# ============================================================

ajami_stoi, ajami_itos = load_vocab(
    AJAMI_VOCAB_FILE
)

wolof_stoi, wolof_itos = load_vocab(
    WOLOF_VOCAB_FILE
)


# ============================================================
# TITRE
# ============================================================

print("=" * 70)
print("DIAGNOSTIC MODELE REVERSE")
print("AJAMI -> WOLOF LATIN")
print("=" * 70)

print("Device :", DEVICE)
print("Model  :", MODEL_PATH)


# ============================================================
# VOCABULAIRES
# ============================================================

print("\n" + "=" * 70)
print("VOCABULAIRES")
print("=" * 70)

print("Vocabulaire Ajami :", len(ajami_stoi))
print("Vocabulaire Wolof :", len(wolof_stoi))


# ============================================================
# TOKENS
# ============================================================

AJAMI_PAD_IDX = ajami_stoi["<PAD>"]
AJAMI_SOS_IDX = ajami_stoi["<SOS>"]
AJAMI_EOS_IDX = ajami_stoi["<EOS>"]
AJAMI_UNK_IDX = ajami_stoi["<UNK>"]

WOLOF_PAD_IDX = wolof_stoi["<PAD>"]
WOLOF_SOS_IDX = wolof_stoi["<SOS>"]
WOLOF_EOS_IDX = wolof_stoi["<EOS>"]
WOLOF_UNK_IDX = wolof_stoi["<UNK>"]


print("\n" + "=" * 70)
print("TOKENS")
print("=" * 70)

print("AJAMI")
print("  PAD :", AJAMI_PAD_IDX)
print("  SOS :", AJAMI_SOS_IDX)
print("  EOS :", AJAMI_EOS_IDX)
print("  UNK :", AJAMI_UNK_IDX)

print()

print("WOLOF")
print("  PAD :", WOLOF_PAD_IDX)
print("  SOS :", WOLOF_SOS_IDX)
print("  EOS :", WOLOF_EOS_IDX)
print("  UNK :", WOLOF_UNK_IDX)


# ============================================================
# CONSTRUCTION DU MODELE
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUCTION DU MODELE REVERSE")
print("=" * 70)

INPUT_DIM = len(ajami_stoi)
OUTPUT_DIM = len(wolof_stoi)


encoder = EncoderGRU(
    input_dim=INPUT_DIM,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    pad_idx=AJAMI_PAD_IDX
)


attention = BahdanauAttention(
    hidden_dim=HIDDEN_DIM
)


decoder = DecoderGRU(
    output_dim=OUTPUT_DIM,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
)


model = Seq2Seq(
    encoder=encoder,
    decoder=decoder,
    attention=attention,
    device=DEVICE,
    src_pad_idx=AJAMI_PAD_IDX
).to(DEVICE)


print("Modele reverse construit.")


# ============================================================
# CHARGEMENT DU CHECKPOINT
# ============================================================

print("\n" + "=" * 70)
print("CHARGEMENT DE BEST_MODEL_REVERSE")
print("=" * 70)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ============================================================
# RECUPERATION DU STATE DICT
# ============================================================

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    state_dict = checkpoint["model_state_dict"]

    print(
        "Epoch :",
        checkpoint.get("epoch", "?")
    )

    print(
        "Loss validation :",
        checkpoint.get("val_loss", "?")
    )

else:

    state_dict = checkpoint

    print("Checkpoint chargé directement.")


# ============================================================
# COMPATIBILITE ANCIEN / NOUVEAU ENCODEUR
# ============================================================
#
# Ancien checkpoint :
#
# encoder.rnn.weight_ih_l0
# encoder.rnn.weight_hh_l0
# encoder.rnn.bias_ih_l0
# encoder.rnn.bias_hh_l0
#
# Nouveau encoder.py :
#
# encoder.gru.weight_ih_l0
# encoder.gru.weight_hh_l0
# encoder.gru.bias_ih_l0
# encoder.gru.bias_hh_l0
#
# L'architecture est identique.
# Seul le nom de l'attribut a changé.
# ============================================================

converted_state_dict = {}

converted_count = 0

for key, value in state_dict.items():

    new_key = key

    if key.startswith("encoder.rnn."):
        new_key = key.replace(
            "encoder.rnn.",
            "encoder.gru.",
            1
        )

        converted_count += 1

    converted_state_dict[new_key] = value


print(
    "Parametres encoder.rnn -> encoder.gru convertis :",
    converted_count
)


# ============================================================
# VERIFICATION DES CLES
# ============================================================

model_keys = set(
    model.state_dict().keys()
)

checkpoint_keys = set(
    converted_state_dict.keys()
)


missing_keys = model_keys - checkpoint_keys
unexpected_keys = checkpoint_keys - model_keys


if missing_keys:

    print("\nATTENTION : clés manquantes")

    for key in sorted(missing_keys):
        print("  ", key)


if unexpected_keys:

    print("\nATTENTION : clés inattendues")

    for key in sorted(unexpected_keys):
        print("  ", key)


# ============================================================
# CHARGEMENT
# ============================================================

model.load_state_dict(
    converted_state_dict,
    strict=True
)

model.eval()


print("\nModele reverse chargé avec succès !")


# ============================================================
# ENCODAGE AJAMI
# ============================================================

def encode_ajami(text):

    ids = []

    # SOS
    ids.append(
        AJAMI_SOS_IDX
    )

    # Caractères Ajami
    for char in text:

        idx = ajami_stoi.get(
            char,
            AJAMI_UNK_IDX
        )

        ids.append(idx)

    # EOS
    ids.append(
        AJAMI_EOS_IDX
    )

    tensor = torch.tensor(
        ids,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    return tensor


# ============================================================
# DECODAGE WOLOF
# ============================================================

def decode_wolof(indices):

    chars = []

    for indice in indices:

        indice = int(indice)

        if (
            0 <= indice
            < len(wolof_itos)
        ):
            caractere = wolof_itos[indice]

        else:
            caractere = "<UNK>"

        # Tokens spéciaux
        if caractere in [
            "<PAD>",
            "<SOS>"
        ]:
            continue

        if caractere == "<EOS>":
            break

        chars.append(
            caractere
        )

    return "".join(chars)


# ============================================================
# PREDICTION
# ============================================================

def predict(
    text,
    max_len=100
):

    src = encode_ajami(text)

    with torch.no_grad():

        # ----------------------------------------------------
        # ENCODEUR
        # ----------------------------------------------------

        encoder_outputs, hidden = model.encoder(
            src
        )

        # ----------------------------------------------------
        # MASQUE SOURCE
        # ----------------------------------------------------

        src_mask = model.create_src_mask(
            src
        )

        # ----------------------------------------------------
        # PREMIER TOKEN = SOS WOLOF
        # ----------------------------------------------------

        input_token = torch.tensor(
            [WOLOF_SOS_IDX],
            dtype=torch.long,
            device=DEVICE
        )

        predicted_indices = []

        attention_history = []

        # ----------------------------------------------------
        # DECODAGE AUTOREGRESSIF
        # ----------------------------------------------------

        for step in range(max_len):

            # Attention Bahdanau
            context, attention_weights = model.attention(
                hidden,
                encoder_outputs,
                mask=src_mask
            )

            attention_history.append(
                attention_weights.squeeze(0).cpu()
            )

            # Decoder
            output, hidden = model.decoder(
                input_token,
                hidden,
                context
            )

            # Meilleure prédiction
            prediction = output.argmax(
                dim=1
            )

            predicted_idx = prediction.item()

            predicted_indices.append(
                predicted_idx
            )

            # EOS
            if predicted_idx == WOLOF_EOS_IDX:
                break

            # Le token prédit devient
            # l'entrée suivante
            input_token = prediction

    return (
        predicted_indices,
        attention_history
    )


# ============================================================
# DIAGNOSTIC D'UNE PREDICTION
# ============================================================

def diagnostic_prediction(sentence):

    print("\n")
    print("=" * 70)
    print("ENTREE AJAMI")
    print("=" * 70)

    print("Texte :", sentence)

    print(
        "Unicode :",
        sentence.encode(
            "unicode_escape"
        ).decode()
    )


    # --------------------------------------------------------
    # ENCODAGE
    # --------------------------------------------------------

    src = encode_ajami(sentence)

    print("\nIDs source :")
    print(
        src.squeeze(0).tolist()
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predicted_indices, attention_history = predict(
        sentence,
        max_len=100
    )


    # --------------------------------------------------------
    # INDICES
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("INDICES PREDITS")
    print("-" * 70)

    print(
        predicted_indices
    )


    # --------------------------------------------------------
    # DETAIL
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("DETAIL DES PREDICTIONS")
    print("-" * 70)

    for position, indice in enumerate(
        predicted_indices
    ):

        indice = int(indice)

        if (
            0 <= indice
            < len(wolof_itos)
        ):
            caractere = wolof_itos[indice]

        else:
            caractere = "<UNK>"

        print(
            f"Position {position:03d} | "
            f"Indice {indice:03d} | "
            f"Caractère {repr(caractere)}"
        )


    # --------------------------------------------------------
    # RESULTAT
    # --------------------------------------------------------

    prediction = decode_wolof(
        predicted_indices
    )

    print("\n" + "-" * 70)
    print("PREDICTION FINALE")
    print("-" * 70)

    print(
        "Résultat :",
        prediction
    )

    print(
        "Unicode  :",
        prediction.encode(
            "unicode_escape"
        ).decode()
    )

    return prediction


# ============================================================
# TESTS CIBLES
# ============================================================

tests = [

    # ndank
    "ندانك",

    # xam
    "خام",

    # nit
    "نيت",

    # jàng
    "جاڠ",

]


# ============================================================
# EXECUTION DES TESTS
# ============================================================

print("\n" + "=" * 70)
print("TESTS CIBLES")
print("=" * 70)


for i, sentence in enumerate(
    tests,
    start=1
):

    print(
        f"\nTEST {i}/{len(tests)}"
    )

    diagnostic_prediction(
        sentence
    )


# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC TERMINE")
print("=" * 70)