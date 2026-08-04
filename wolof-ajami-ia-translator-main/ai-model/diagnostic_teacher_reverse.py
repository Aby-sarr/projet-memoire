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
print("DIAGNOSTIC TEACHER FORCING - MODELE REVERSE")
print("AJAMI -> WOLOF LATIN")
print("=" * 70)

print("Device :", DEVICE)
print("Seed :", SEED)


# ============================================================
# 1. CHARGEMENT DES VOCABULAIRES
# ============================================================

print()
print("=" * 70)
print("CHARGEMENT DES VOCABULAIRES")
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


# ------------------------------------------------------------
# SOURCE = AJAMI
# ------------------------------------------------------------

ajami_stoi = ajami_vocab["stoi"]
ajami_itos = ajami_vocab["itos"]


# ------------------------------------------------------------
# CIBLE = WOLOF LATIN
# ------------------------------------------------------------

wolof_stoi = wolof_vocab["stoi"]
wolof_itos = wolof_vocab["itos"]


# ============================================================
# 2. SECURITE ITOS
# ============================================================

# Si itos est un dictionnaire, on convertit les clés
# en entiers.
#
# Si itos est déjà une liste, on la conserve telle quelle.

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
# 3. FONCTION SECURISEE POUR LIRE ITOS
# ============================================================

def get_char(itos, index):

    index = int(index)

    # --------------------------------------------------------
    # Cas 1 : itos = liste
    # --------------------------------------------------------

    if isinstance(itos, list):

        if (
            index < 0
            or index >= len(itos)
        ):

            return "<INVALID>"

        return itos[index]

    # --------------------------------------------------------
    # Cas 2 : itos = dictionnaire
    # --------------------------------------------------------

    if isinstance(itos, dict):

        return itos.get(
            index,
            "<INVALID>"
        )

    return "<INVALID>"


# ============================================================
# 4. VERIFICATION DES TOKENS
# ============================================================

print()
print("=" * 70)
print("VERIFICATION DES TOKENS")
print("=" * 70)


required_tokens = [
    "<PAD>",
    "<SOS>",
    "<EOS>",
    "<UNK>",
]


for token in required_tokens:

    if token not in ajami_stoi:

        raise ValueError(
            f"Token Ajami manquant : {token}"
        )


for token in required_tokens:

    if token not in wolof_stoi:

        raise ValueError(
            f"Token Wolof manquant : {token}"
        )


print("Tokens Ajami : OK")
print("Tokens Wolof : OK")


# ============================================================
# 5. VERIFICATION DES INDICES IMPORTANTS
# ============================================================

print()
print("=" * 70)
print("VERIFICATION DES INDICES WOLOF")
print("=" * 70)


for char in [
    "C",
    "c",
    "a",
    "d",
    "n",
]:

    if char in wolof_stoi:

        index = wolof_stoi[char]

        print(
            f"{char} = {index}"
        )

    else:

        print(
            f"{char} = ABSENT"
        )


print(
    "SOS =",
    wolof_stoi["<SOS>"]
)

print(
    "EOS =",
    wolof_stoi["<EOS>"]
)

print(
    "UNK =",
    wolof_stoi["<UNK>"]
)


# ============================================================
# 6. CONSTRUCTION DU MODELE
# ============================================================

print()
print("=" * 70)
print("CONSTRUCTION DU MODELE REVERSE")
print("=" * 70)


# SOURCE = AJAMI

encoder = EncoderGRU(
    input_dim=len(ajami_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)


# ATTENTION

attention = BahdanauAttention(
    HIDDEN_SIZE
)


# CIBLE = WOLOF LATIN

decoder = DecoderGRU(
    output_dim=len(wolof_stoi),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)


# MODELE COMPLET

model = Seq2Seq(
    encoder,
    decoder,
    attention,
    DEVICE
).to(DEVICE)


print(
    "Modèle créé avec succès."
)


# ============================================================
# 7. CHARGEMENT DU MODELE REVERSE
# ============================================================

MODEL_PATH = (
    MODEL_DIR / "best_model_reverse.pt"
)


print()
print("=" * 70)
print("CHARGEMENT DE BEST_MODEL_REVERSE")
print("=" * 70)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Modèle introuvable : {MODEL_PATH}"
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

    print(
        "Modèle chargé avec succès."
    )

    if "epoch" in checkpoint:

        print(
            "Epoch :",
            checkpoint["epoch"]
        )

    if "valid_loss" in checkpoint:

        print(
            "Validation loss :",
            checkpoint["valid_loss"]
        )

    elif "val_loss" in checkpoint:

        print(
            "Validation loss :",
            checkpoint["val_loss"]
        )

else:

    model.load_state_dict(
        checkpoint
    )

    print(
        "State dict chargé directement."
    )


model.eval()


# ============================================================
# 8. ENCODAGE AJAMI
# ============================================================

def encode_ajami(text):

    ids = []


    for char in str(text):

        if char in ajami_stoi:

            ids.append(
                ajami_stoi[char]
            )

        else:

            ids.append(
                ajami_stoi["<UNK>"]
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
# 9. DIAGNOSTIC TEACHER FORCING
# ============================================================

def diagnostic_teacher(
    ajami_text,
    wolof_text
):

    print()
    print("=" * 70)
    print("NOUVEAU TEST")
    print("=" * 70)

    print()
    print(
        "Ajami :",
        repr(ajami_text)
    )

    print(
        "Wolof attendu :",
        repr(wolof_text)
    )


    # ========================================================
    # SOURCE AJAMI
    # ========================================================

    source = encode_ajami(
        ajami_text
    )


    print()
    print(
        "Indices source :"
    )

    print(
        source.squeeze(0).tolist()
    )


    # ========================================================
    # CIBLE WOLOF
    # ========================================================

    target_ids = []


    for char in wolof_text:

        if char in wolof_stoi:

            target_ids.append(
                wolof_stoi[char]
            )

        else:

            target_ids.append(
                wolof_stoi["<UNK>"]
            )


    target_ids = (
        [wolof_stoi["<SOS>"]]
        + target_ids
        + [wolof_stoi["<EOS>"]]
    )


    target = torch.tensor(
        target_ids,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)


    print()
    print(
        "Indices cible :"
    )

    print(
        target.squeeze(0).tolist()
    )


    # ========================================================
    # ENCODAGE
    # ========================================================

    with torch.no_grad():

        encoder_outputs, hidden = (
            model.encoder(source)
        )


    # ========================================================
    # PREMIER TOKEN
    # ========================================================

    input_token = target[:, 0]


    print()
    print("=" * 70)
    print("PREDICTIONS AVEC VRAIS TOKENS")
    print("=" * 70)


    correct = 0
    total = 0


    # ========================================================
    # DECODAGE
    # ========================================================

    for step in range(
        1,
        target.shape[1]
    ):

        with torch.no_grad():

            # ------------------------------------------------
            # Attention
            # ------------------------------------------------

            context, attention_weights = (
                model.attention(
                    hidden,
                    encoder_outputs
                )
            )


            # ------------------------------------------------
            # Décodeur
            # ------------------------------------------------

            output, hidden = (
                model.decoder(
                    input_token,
                    hidden,
                    context
                )
            )


        # ====================================================
        # PROBABILITES
        # ====================================================

        probabilities = torch.softmax(
            output,
            dim=1
        )


        predicted_token = (
            output.argmax(
                dim=1
            ).item()
        )


        probability = (
            probabilities[
                0,
                predicted_token
            ].item()
        )


        # ====================================================
        # CIBLE REELLE
        # ====================================================

        target_token = (
            target[
                0,
                step
            ].item()
        )


        predicted_char = get_char(
            wolof_itos,
            predicted_token
        )


        target_char = get_char(
            wolof_itos,
            target_token
        )


        input_char = get_char(
            wolof_itos,
            input_token.item()
        )


        # ====================================================
        # COMPARAISON
        # ====================================================

        is_correct = (
            predicted_token
            == target_token
        )


        if target_char not in [
            "<EOS>",
            "<PAD>",
            "<SOS>",
        ]:

            total += 1

            if is_correct:

                correct += 1


        status = (
            "OK"
            if is_correct
            else "ERREUR"
        )


        # ====================================================
        # AFFICHAGE
        # ====================================================

        print()
        print(
            f"Etape {step}"
        )


        print(
            "Entrée décodeur :",
            repr(input_char)
        )


        print(
            "Cible réelle     :",
            repr(target_char),
            f"(indice {target_token})"
        )


        print(
            "Prédiction       :",
            repr(predicted_char),
            f"(indice {predicted_token})"
        )


        print(
            "Probabilité      :",
            f"{probability:.6f}"
        )


        print(
            "Résultat         :",
            status
        )


        # ====================================================
        # TOP 5
        # ====================================================

        top_probs, top_indices = torch.topk(
            probabilities[0],
            k=5
        )


        print()
        print(
            "Top 5 :"
        )


        for rank in range(5):

            idx = (
                top_indices[
                    rank
                ].item()
            )


            prob = (
                top_probs[
                    rank
                ].item()
            )


            char = get_char(
                wolof_itos,
                idx
            )


            print(
                f"  {rank + 1}. "
                f"{repr(char)} "
                f"(indice={idx}, "
                f"p={prob:.6f})"
            )


        # ====================================================
        # ATTENTION
        # ====================================================

        attention_position = (
            attention_weights.argmax(
                dim=1
            ).item()
        )


        attention_value = (
            attention_weights[
                0,
                attention_position
            ].item()
        )


        print()
        print(
            "Attention max :",
            f"{attention_value:.6f}",
            "position",
            attention_position
        )


        # Affichage de toute l'attention

        print(
            "Attention complète :"
        )


        attention_values = (
            attention_weights[
                0
            ].tolist()
        )


        for position, value in enumerate(
            attention_values
        ):

            source_index = (
                source[
                    0,
                    position
                ].item()
            )


            source_char = get_char(
                ajami_itos,
                source_index
            )


            print(
                f"  Position {position:02d} "
                f"| {repr(source_char):>8} "
                f"| {value:.6f}"
            )


        # ====================================================
        # TEACHER FORCING
        # ====================================================
        #
        # C'est le point essentiel de ce diagnostic.
        #
        # Au lieu de donner la prédiction du modèle comme
        # entrée suivante, nous donnons le VRAI caractère
        # cible.
        #
        # Exemple :
        #
        # <SOS> -> n
        # n     -> d
        # d     -> a
        # a     -> n
        # n     -> k
        #
        # Cela permet de savoir si le modèle connaît réellement
        # la bonne sortie à chaque position.
        # ====================================================

        input_token = target[
            :,
            step
        ]


    # ========================================================
    # RESULTAT
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTAT DU TEST")
    print("=" * 70)


    if total > 0:

        accuracy = (
            100.0
            * correct
            / total
        )

    else:

        accuracy = 0.0


    print(
        f"Caractères corrects : "
        f"{correct}/{total}"
    )


    print(
        f"Accuracy teacher forcing : "
        f"{accuracy:.2f} %"
    )


# ============================================================
# 10. TESTS
# ============================================================

tests = [

    (
        "ندانك",
        "ndank"
    ),

    (
        "مووي",
        "mooy"
    ),

    (
        "جاپپ",
        "japp"
    ),

    (
        "گولو",
        "golo"
    ),

    (
        "چي",
        "ci"
    ),

    (
        "ڽااي",
        "ñaay"
    ),

    (
        "يومب",
        "yomb"
    ),

    (
        "دێگگ",
        "dëgg"
    ),

    (
        "نيت",
        "nit"
    ),

    (
        "كو",
        "ku"
    ),
]


# ============================================================
# 11. EXECUTION
# ============================================================

print()
print("=" * 70)
print("DEBUT DES TESTS TEACHER FORCING")
print("=" * 70)


for ajami, wolof in tests:

    diagnostic_teacher(
        ajami,
        wolof
    )


# ============================================================
# FIN
# ============================================================

print()
print("=" * 70)
print("FIN DU DIAGNOSTIC")
print("=" * 70)