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
# CONFIGURATION
# ============================================================

SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 70)
print("DIAGNOSTIC LOGITS - MODELE REVERSE")
print("AJAMI -> WOLOF LATIN")
print("=" * 70)

print("Device :", DEVICE)


# ============================================================
# VOCABULAIRES
# ============================================================

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


ajami_stoi = ajami_vocab["stoi"]
ajami_itos = ajami_vocab["itos"]

wolof_stoi = wolof_vocab["stoi"]
wolof_itos = wolof_vocab["itos"]


if isinstance(ajami_itos, dict):
    ajami_itos = {
        int(k): v
        for k, v in ajami_itos.items()
    }


if isinstance(wolof_itos, dict):
    wolof_itos = {
        int(k): v
        for k, v in wolof_itos.items()
    }


print()
print("Ajami vocab :", len(ajami_stoi))
print("Wolof vocab :", len(wolof_stoi))


# ============================================================
# MODELE
# ============================================================

encoder = EncoderGRU(
    input_dim=len(ajami_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)


attention = BahdanauAttention(
    HIDDEN_SIZE
)


decoder = DecoderGRU(
    output_dim=len(wolof_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)


model = Seq2Seq(
    encoder,
    decoder,
    attention,
    DEVICE
).to(DEVICE)


# ============================================================
# CHARGEMENT
# ============================================================

MODEL_PATH = (
    MODEL_DIR / "best_model_reverse.pt"
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()


print()
print("=" * 70)
print("MODELE CHARGE")
print("=" * 70)

if isinstance(checkpoint, dict):

    print(
        "Epoch :",
        checkpoint.get("epoch", "?")
    )

    print(
        "Validation loss :",
        checkpoint.get(
            "valid_loss",
            checkpoint.get("val_loss", "?")
        )
    )


# ============================================================
# ENCODAGE
# ============================================================

def encode_ajami(text):

    ids = []

    for char in text:

        ids.append(
            ajami_stoi.get(
                char,
                ajami_stoi["<UNK>"]
            )
        )


    ids = (
        [ajami_stoi["<SOS>"]]
        + ids
        + [ajami_stoi["<EOS>"]]
    )


    return torch.tensor(
        ids,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)


# ============================================================
# TEST
# ============================================================

text = "گوددي"

print()
print("=" * 70)
print("ENTREE")
print("=" * 70)

print("Ajami :", text)

source = encode_ajami(text)

print(
    "Indices source :",
    source.squeeze(0).tolist()
)


# ============================================================
# ENCODER
# ============================================================

with torch.no_grad():

    encoder_outputs, hidden = (
        model.encoder(source)
    )


# ============================================================
# PREMIER TOKEN
# ============================================================

input_token = torch.tensor(
    [wolof_stoi["<SOS>"]],
    dtype=torch.long,
    device=DEVICE
)


print()
print("=" * 70)
print("PREMIERE PREDICTION")
print("=" * 70)

print(
    "Token entrée décodeur :",
    wolof_stoi["<SOS>"]
)


with torch.no_grad():

    context, attention_weights = (
        model.attention(
            hidden,
            encoder_outputs
        )
    )


    output, hidden2 = (
        model.decoder(
            input_token,
            hidden,
            context
        )
    )


# ============================================================
# TOP 10
# ============================================================

probabilities = torch.softmax(
    output,
    dim=1
)


top_values, top_indices = torch.topk(
    probabilities,
    k=10,
    dim=1
)


print()
print("=" * 70)
print("TOP 10 DES PREDICTIONS")
print("=" * 70)


for rank in range(10):

    index = top_indices[
        0,
        rank
    ].item()


    probability = top_values[
        0,
        rank
    ].item()


    if 0 <= index < len(wolof_itos):

        char = wolof_itos[index]

    else:

        char = "<INVALID>"


    print(
        f"{rank + 1:02d}. "
        f"indice={index:02d} "
        f"caractere={repr(char):12} "
        f"probabilite={probability:.6f}"
    )


# ============================================================
# INFORMATION SPECIFIQUE SUR C
# ============================================================

c_index = wolof_stoi["C"]

c_probability = probabilities[
    0,
    c_index
].item()


print()
print("=" * 70)
print("DIAGNOSTIC DU CARACTERE C")
print("=" * 70)

print(
    "Indice C :",
    c_index
)

print(
    "Probabilité de C à la première étape :",
    f"{c_probability:.6f}"
)


# ============================================================
# EOS
# ============================================================

eos_index = wolof_stoi["<EOS>"]

eos_probability = probabilities[
    0,
    eos_index
].item()


print(
    "Indice EOS :",
    eos_index
)

print(
    "Probabilité EOS :",
    f"{eos_probability:.6f}"
)


# ============================================================
# ATTENTION
# ============================================================

print()
print("=" * 70)
print("ATTENTION")
print("=" * 70)

weights = attention_weights[
    0
].cpu().tolist()


for i, weight in enumerate(weights):

    print(
        f"Source position {i:02d} : "
        f"{weight:.6f}"
    )


print()
print("=" * 70)
print("FIN DU DIAGNOSTIC")
print("=" * 70)