import torch
import json
import pandas as pd

from config.config import (
    WOLOF_VOCAB_FILE,
    AJAMI_VOCAB_FILE,
    MODEL_DIR,
    CORPUS_FILE,
    EMBEDDING,
    HIDDEN_SIZE,
    DEVICE,
)

from models.encoder import EncoderGRU
from models.attention import BahdanauAttention
from models.decoder import DecoderGRU
from models.seq2seq import Seq2Seq


# ==================================================
# 1. CHARGEMENT DES VOCABULAIRES
# ==================================================

print("=" * 50)
print("Chargement des vocabulaires...")
print("=" * 50)

with open(WOLOF_VOCAB_FILE, "r", encoding="utf-8") as f:
    wolof_vocab = json.load(f)

with open(AJAMI_VOCAB_FILE, "r", encoding="utf-8") as f:
    ajami_vocab = json.load(f)

wolof_stoi = wolof_vocab["stoi"]
ajami_itos = ajami_vocab["itos"]

print("Vocabulaire Wolof :", len(wolof_stoi))
print("Vocabulaire Ajami :", len(ajami_itos))


# ==================================================
# 2. CONSTRUCTION DU MODELE
# ==================================================

print("=" * 50)
print("Construction du modèle...")
print("=" * 50)

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


# ==================================================
# 3. CHARGEMENT DU MEILLEUR MODELE
# ==================================================

MODEL_PATH = MODEL_DIR / "best_model.pt"

print("=" * 50)
print("Chargement du meilleur modèle...")
print("=" * 50)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Modèle chargé avec succès !")
print("Epoch du meilleur modèle :", checkpoint["epoch"])
print("Loss validation :", checkpoint["valid_loss"])


# ==================================================
# 4. CHARGEMENT DU CORPUS COMPLET
# ==================================================

print("=" * 50)
print("Chargement du corpus...")
print("=" * 50)

df = pd.read_csv(
    CORPUS_FILE
)

print("Nombre total de phrases :", len(df))


# ==================================================
# 5. FONCTION DE PREDICTION
# ==================================================

def predict(sentence):

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


    # --------------------------------------------------
    # Longueur maximale adaptée à la phrase
    # --------------------------------------------------

    MAX_LENGTH = len(sentence) + 10

    pad_idx = ajami_itos.index("<PAD>")
    sos_idx = ajami_itos.index("<SOS>")

    target = torch.full(
        (1, MAX_LENGTH),
        pad_idx,
        dtype=torch.long,
        device=DEVICE
    )

    target[0, 0] = sos_idx


    with torch.no_grad():

        output = model(
            source,
            target,
            teacher_forcing_ratio=0
        )


    predictions = output.argmax(
        dim=2
    )


    resultat = []

    for indice in predictions[0]:

        caractere = ajami_itos[
            indice.item()
        ]

        if caractere == "<EOS>":
            break

        if caractere in [
            "<PAD>",
            "<SOS>"
        ]:
            continue

        resultat.append(
            caractere
        )

    return "".join(resultat)


# ==================================================
# 6. DISTANCE DE LEVENSHTEIN
# ==================================================

def levenshtein(reference, prediction):

    n = len(reference)
    m = len(prediction)

    matrix = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]


    for i in range(n + 1):
        matrix[i][0] = i


    for j in range(m + 1):
        matrix[0][j] = j


    for i in range(1, n + 1):

        for j in range(1, m + 1):

            if reference[i - 1] == prediction[j - 1]:

                cost = 0

            else:

                cost = 1


            matrix[i][j] = min(

                matrix[i - 1][j] + 1,

                matrix[i][j - 1] + 1,

                matrix[i - 1][j - 1] + cost

            )


    return matrix[n][m]


# ==================================================
# 7. DEBUT DE L'EVALUATION
# ==================================================

print("=" * 50)
print("Début de l'évaluation complète...")
print("=" * 50)


total_correct = 0
total_sentences = 0

total_char_errors = 0
total_reference_chars = 0

total_word_errors = 0
total_reference_words = 0

errors = []


# ==================================================
# 8. PARCOURS DES 20 562 PHRASES
# ==================================================

for index, row in df.iterrows():


    # --------------------------------------------------
    # Affichage de la progression
    # --------------------------------------------------

    if index % 100 == 0:

        print(
            f"Évaluation : "
            f"{index}/{len(df)} phrases..."
        )


    # --------------------------------------------------
    # Données
    # --------------------------------------------------

    wolof = str(
        row["Wolof"]
    )

    reference = str(
        row["ajami"]
    )


    # --------------------------------------------------
    # Prédiction
    # --------------------------------------------------

    prediction = predict(
        wolof
    )


    # ==================================================
    # ACCURACY
    # ==================================================

    if prediction == reference:

        total_correct += 1


    total_sentences += 1


    # ==================================================
    # CER
    # ==================================================

    char_distance = levenshtein(
        reference,
        prediction
    )

    total_char_errors += char_distance

    total_reference_chars += len(
        reference
    )


    # ==================================================
    # WER
    # ==================================================

    reference_words = reference.split()

    prediction_words = prediction.split()


    word_distance = levenshtein(
        reference_words,
        prediction_words
    )


    total_word_errors += word_distance

    total_reference_words += len(
        reference_words
    )


    # ==================================================
    # STOCKAGE DES ERREURS
    # ==================================================

    if prediction != reference:

        errors.append(
            (
                wolof,
                reference,
                prediction
            )
        )


# ==================================================
# 9. CALCUL ACCURACY
# ==================================================

if total_sentences > 0:

    accuracy = (
        total_correct
        / total_sentences
    ) * 100

else:

    accuracy = 0


# ==================================================
# 10. CALCUL CER
# ==================================================

if total_reference_chars > 0:

    cer = (
        total_char_errors
        / total_reference_chars
    ) * 100

else:

    cer = 0


# ==================================================
# 11. CALCUL WER
# ==================================================

if total_reference_words > 0:

    wer = (
        total_word_errors
        / total_reference_words
    ) * 100

else:

    wer = 0


# ==================================================
# 12. RESULTATS FINAUX
# ==================================================

print()
print("=" * 50)
print("RESULTATS FINAUX")
print("=" * 50)

print(
    "Nombre total de phrases :",
    total_sentences
)

print(
    "Phrases correctes :",
    total_correct
)

print(
    f"Accuracy : {accuracy:.2f}%"
)

print(
    "Erreurs caractères :",
    total_char_errors
)

print(
    f"CER : {cer:.2f}%"
)

print(
    "Erreurs mots :",
    total_word_errors
)

print(
    f"WER : {wer:.2f}%"
)


# ==================================================
# 13. EXEMPLES D'ERREURS
# ==================================================

print()
print("=" * 50)
print("EXEMPLES D'ERREURS")
print("=" * 50)


if len(errors) == 0:

    print(
        "Aucune erreur !"
    )

else:

    for i, (
        wolof,
        reference,
        prediction
    ) in enumerate(
        errors[:20],
        start=1
    ):

        print()
        print(
            f"Erreur {i}"
        )

        print(
            "Wolof    :",
            wolof
        )

        print(
            "Attendu  :",
            reference
        )

        print(
            "Prédit   :",
            prediction
        )


# ==================================================
# 14. FIN
# ==================================================

print()
print("=" * 50)
print("Evaluation complète terminée.")
print("=" * 50)