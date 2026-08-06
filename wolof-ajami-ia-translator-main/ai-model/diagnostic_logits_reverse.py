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
from models.decoder import DecoderGRU
from models.attention import BahdanauAttention
from models.seq2seq import Seq2Seq


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
MAX_LENGTH = 30
TOP_K = 8

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


MODEL_PATH = MODEL_DIR / "best_model_reverse.pt"


# ============================================================
# TITRE
# ============================================================

print("=" * 70)
print("DIAGNOSTIC LOGITS - MODELE REVERSE")
print("AJAMI -> WOLOF LATIN")
print("=" * 70)

print("Device :", DEVICE)
print("Model  :", MODEL_PATH)


# ============================================================
# CHARGEMENT VOCABULAIRE
# ============================================================

def load_vocab(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    stoi = data["stoi"]
    itos = data["itos"]

    # Certains fichiers ont itos sous forme de dictionnaire.
    if isinstance(itos, dict):

        itos = {
            int(k): v
            for k, v in itos.items()
        }

    return stoi, itos


ajami_stoi, ajami_itos = load_vocab(
    AJAMI_VOCAB_FILE
)

wolof_stoi, wolof_itos = load_vocab(
    WOLOF_VOCAB_FILE
)


# ============================================================
# FONCTION DE LECTURE itos
# ============================================================

def get_token(itos, index):

    index = int(index)

    if isinstance(itos, list):

        if 0 <= index < len(itos):

            return itos[index]

        return "<UNK>"

    if isinstance(itos, dict):

        return itos.get(
            index,
            "<UNK>"
        )

    return "<UNK>"


# ============================================================
# INFORMATIONS VOCABULAIRES
# ============================================================

print()
print("=" * 70)
print("VOCABULAIRES")
print("=" * 70)

print(
    "Vocabulaire source Ajami :",
    len(ajami_stoi)
)

print(
    "Vocabulaire cible Wolof :",
    len(wolof_stoi)
)


# ============================================================
# TOKENS
# ============================================================

AJAMI_PAD = ajami_stoi["<PAD>"]
AJAMI_SOS = ajami_stoi["<SOS>"]
AJAMI_EOS = ajami_stoi["<EOS>"]
AJAMI_UNK = ajami_stoi["<UNK>"]

WOLOF_PAD = wolof_stoi["<PAD>"]
WOLOF_SOS = wolof_stoi["<SOS>"]
WOLOF_EOS = wolof_stoi["<EOS>"]
WOLOF_UNK = wolof_stoi["<UNK>"]


print()
print("=" * 70)
print("TOKENS")
print("=" * 70)

print("AJAMI")
print("  PAD :", AJAMI_PAD)
print("  SOS :", AJAMI_SOS)
print("  EOS :", AJAMI_EOS)
print("  UNK :", AJAMI_UNK)

print()

print("WOLOF")
print("  PAD :", WOLOF_PAD)
print("  SOS :", WOLOF_SOS)
print("  EOS :", WOLOF_EOS)
print("  UNK :", WOLOF_UNK)


# ============================================================
# CONSTRUCTION MODELE
# ============================================================

print()
print("=" * 70)
print("CONSTRUCTION DU MODELE REVERSE")
print("=" * 70)


encoder = EncoderGRU(
    input_dim=len(ajami_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE,
    pad_idx=AJAMI_PAD
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
    encoder=encoder,
    decoder=decoder,
    attention=attention,
    device=DEVICE,
    src_pad_idx=AJAMI_PAD
).to(DEVICE)


print("Modele reverse construit.")


# ============================================================
# CHARGEMENT CHECKPOINT
# ============================================================

print()
print("=" * 70)
print("CHARGEMENT DE BEST_MODEL_REVERSE")
print("=" * 70)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    state_dict = checkpoint[
        "model_state_dict"
    ]

    print(
        "Epoch :",
        checkpoint.get(
            "epoch",
            "?"
        )
    )

    print(
        "Valid loss :",
        checkpoint.get(
            "valid_loss",
            "?"
        )
    )

else:

    state_dict = checkpoint

    print(
        "Checkpoint state_dict direct."
    )


# ============================================================
# COMPATIBILITE ANCIEN NOM encoder.rnn
# ============================================================

converted = 0

new_state_dict = {}

for key, value in state_dict.items():

    new_key = key.replace(
        "encoder.rnn.",
        "encoder.gru."
    )

    if new_key != key:

        converted += 1

    new_state_dict[
        new_key
    ] = value


print()
print(
    "Parametres encoder.rnn -> "
    f"encoder.gru convertis : {converted}"
)


model.load_state_dict(
    new_state_dict
)

model.eval()

print(
    "Modele reverse charge avec succes."
)


# ============================================================
# ENCODAGE AJAMI
# ============================================================

def encode_ajami(text):

    ids = []

    for char in text:

        idx = ajami_stoi.get(
            char,
            AJAMI_UNK
        )

        ids.append(idx)


    # IMPORTANT :
    # Le pipeline reverse utilise :
    #
    # Ajami + EOS
    #
    # et non :
    #
    # SOS + Ajami + EOS

    source_indices = (
        ids
        + [AJAMI_EOS]
    )


    source = torch.tensor(
        source_indices,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)


    return source


# ============================================================
# DECODAGE D'UN INDICE
# ============================================================

def decode_index(index):

    return get_token(
        wolof_itos,
        index
    )


# ============================================================
# TOP-K LOGITS
# ============================================================

def show_top_predictions(
    logits,
    step
):

    probabilities = torch.softmax(
        logits,
        dim=1
    )


    top_values, top_indices = (
        torch.topk(
            probabilities,
            k=min(
                TOP_K,
                probabilities.size(1)
            ),
            dim=1
        )
    )


    print()
    print(
        f"Etape {step:02d}"
    )

    print(
        "-" * 70
    )


    for rank in range(
        top_indices.size(1)
    ):

        idx = top_indices[
            0,
            rank
        ].item()

        probability = top_values[
            0,
            rank
        ].item()


        token = decode_index(
            idx
        )


        print(
            f"{rank + 1:02d}. "
            f"indice={idx:03d} "
            f"token={repr(token):12s} "
            f"prob={probability:.6f}"
        )


# ============================================================
# DIAGNOSTIC D'UNE SEQUENCE
# ============================================================

def diagnostic_sentence(
    sentence,
    reference=None
):

    print()
    print()
    print("=" * 70)
    print("ENTREE")
    print("=" * 70)

    print(
        "Ajami :",
        sentence
    )

    print(
        "Unicode :",
        sentence.encode(
            "unicode_escape"
        ).decode()
    )


    if reference is not None:

        print(
            "Reference :",
            reference
        )

        print(
            "Unicode ref.:",
            reference.encode(
                "unicode_escape"
            ).decode()
        )


    # --------------------------------------------------------
    # ENCODAGE
    # --------------------------------------------------------

    source = encode_ajami(
        sentence
    )


    print()
    print(
        "IDs source :"
    )

    print(
        source[0].tolist()
    )


    print()
    print(
        "Tokens source :"
    )


    for position, idx in enumerate(
        source[0].tolist()
    ):

        if idx == AJAMI_EOS:

            token = "<EOS>"

        else:

            # On récupère le caractère directement
            # depuis l'entrée source.

            if position < len(sentence):

                token = sentence[position]

            else:

                token = "<UNK>"


        print(
            f"Position {position:02d} | "
            f"ID {idx:03d} | "
            f"Token {repr(token)}"
        )


    # --------------------------------------------------------
    # ENCODEUR
    # --------------------------------------------------------

    with torch.no_grad():

        encoder_outputs, hidden = (
            model.encoder(source)
        )


    # --------------------------------------------------------
    # MASQUE SOURCE
    # --------------------------------------------------------

    src_mask = (
        source
        != model.src_pad_idx
    )


    # --------------------------------------------------------
    # PREMIER TOKEN CIBLE = SOS
    # --------------------------------------------------------

    input_token = torch.tensor(
        [WOLOF_SOS],
        dtype=torch.long,
        device=DEVICE
    )


    predicted_indices = []

    predicted_tokens = []


    # --------------------------------------------------------
    # DECODAGE AUTOREGRESSIF
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANALYSE DES LOGITS ET DE L'ATTENTION")
    print("=" * 70)


    for step in range(
        MAX_LENGTH
    ):

        with torch.no_grad():

            context, attention_weights = (
                model.attention(
                    hidden,
                    encoder_outputs,
                    mask=src_mask
                )
            )


            output, hidden = (
                model.decoder(
                    input_token,
                    hidden,
                    context
                )
            )


        # ----------------------------------------------------
        # TOP-K
        # ----------------------------------------------------

        show_top_predictions(
            output,
            step
        )


        # ----------------------------------------------------
        # PREDICTION PRINCIPALE
        # ----------------------------------------------------

        prediction = output.argmax(
            dim=1
        )


        predicted_idx = prediction.item()


        predicted_indices.append(
            predicted_idx
        )


        predicted_token = decode_index(
            predicted_idx
        )


        predicted_tokens.append(
            predicted_token
        )


        # ----------------------------------------------------
        # ATTENTION
        # ----------------------------------------------------

        weights = (
            attention_weights[
                0
            ]
            .detach()
            .cpu()
            .tolist()
        )


        print()
        print(
            "Prediction choisie :"
        )

        print(
            f"Indice : {predicted_idx}"
        )

        print(
            f"Token  : {repr(predicted_token)}"
        )


        print()
        print(
            "Attention :"
        )


        for source_pos, weight in enumerate(
            weights
        ):

            if source_pos < len(sentence):

                source_char = (
                    sentence[source_pos]
                )

            else:

                source_char = "<EOS>"


            print(
                f"  source[{source_pos:02d}] "
                f"{repr(source_char):8s} "
                f"poids={weight:.6f}"
            )


        # ----------------------------------------------------
        # EOS
        # ----------------------------------------------------

        if predicted_idx == WOLOF_EOS:

            print()
            print(
                "EOS atteint."
            )

            break


        # ----------------------------------------------------
        # PROCHAIN TOKEN
        # ----------------------------------------------------

        input_token = prediction


    # ========================================================
    # RESULTAT FINAL
    # ========================================================

    output_text = ""

    for idx in predicted_indices:

        token = decode_index(
            idx
        )

        if token in [
            "<PAD>",
            "<SOS>",
            "<UNK>"
        ]:

            continue


        if token == "<EOS>":

            break


        output_text += token


    print()
    print("=" * 70)
    print("RESULTAT FINAL")
    print("=" * 70)

    print(
        "Prediction :",
        output_text
    )

    print(
        "Unicode pred.:",
        output_text.encode(
            "unicode_escape"
        ).decode()
    )


    print()
    print(
        "Indices predits :"
    )

    print(
        predicted_indices
    )


    print()
    print(
        "Tokens predits :"
    )

    print(
        predicted_tokens
    )


    # --------------------------------------------------------
    # COMPARAISON REFERENCE
    # --------------------------------------------------------

    if reference is not None:

        print()
        print(
            "Reference :",
            reference
        )

        print(
            "Prediction :",
            output_text
        )

        print(
            "Exact match :",
            output_text == reference
        )


# ============================================================
# TESTS
# ============================================================

tests = [

    (
        "ن",
        "n"
    ),

    (
        "نيت",
        "nit"
    ),

    (
        "ندانك",
        "ndank"
    ),

    (
        "خام",
        "xam"
    ),

    (
        "جاڠ",
        "jàng"
    ),

]


# ============================================================
# EXECUTION
# ============================================================

print()
print("=" * 70)
print("TESTS DIAGNOSTIQUES")
print("=" * 70)


for i, (
    sentence,
    reference
) in enumerate(
    tests,
    start=1
):

    print()
    print(
        f"TEST {i}/{len(tests)}"
    )

    diagnostic_sentence(
        sentence,
        reference
    )


# ============================================================
# FIN
# ============================================================

print()
print("=" * 70)
print("DIAGNOSTIC LOGITS TERMINE")
print("=" * 70)