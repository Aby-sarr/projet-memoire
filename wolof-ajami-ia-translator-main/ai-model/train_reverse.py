import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import random_split

from config.config import (
    DEVICE,
    BATCH_SIZE,
    EMBEDDING,
    HIDDEN_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    TEACHER_FORCING_RATIO,
    CORPUS_FILE,
    WOLOF_VOCAB_FILE,
    AJAMI_VOCAB_FILE,
    MODEL_DIR,
)

from utils.vocabulary import Vocabulary

from utils.dataset_reverse import (
    WolofAjamiReverseDataset
)

from utils.dataloader_reverse import (
    create_dataloader_reverse
)

from models.encoder import EncoderGRU
from models.attention import BahdanauAttention
from models.decoder import DecoderGRU
from models.seq2seq import Seq2Seq


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

MAX_EPOCHS = NUM_EPOCHS

EARLY_STOPPING_PATIENCE = 3


# ============================================================
# SEED
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 1. CONFIGURATION
# ============================================================

print("=" * 60)
print("ENTRAINEMENT MODELE AJAMI -> WOLOF LATIN")
print("=" * 60)

print("Device :", DEVICE)
print("Batch size :", BATCH_SIZE)
print("Embedding :", EMBEDDING)
print("Hidden size :", HIDDEN_SIZE)
print("Nombre maximum d'epochs :", MAX_EPOCHS)
print("Learning rate :", LEARNING_RATE)
print("Teacher forcing :", TEACHER_FORCING_RATIO)
print(
    "Early stopping patience :",
    EARLY_STOPPING_PATIENCE
)
print("Seed :", SEED)


# ============================================================
# 2. CHARGEMENT DES VOCABULAIRES
# ============================================================

print()
print("=" * 60)
print("CHARGEMENT DES VOCABULAIRES")
print("=" * 60)


# ------------------------------------------------------------
# SOURCE = AJAMI
# ------------------------------------------------------------

source_vocab = Vocabulary(
    AJAMI_VOCAB_FILE
)


# ------------------------------------------------------------
# CIBLE = WOLOF LATIN
# ------------------------------------------------------------

target_vocab = Vocabulary(
    WOLOF_VOCAB_FILE
)


print(
    "Vocabulaire source Ajami :",
    len(source_vocab)
)

print(
    "Vocabulaire cible Wolof :",
    len(target_vocab)
)


# ============================================================
# 3. VERIFICATION DES TOKENS
# ============================================================

print()
print("=" * 60)
print("VERIFICATION DES TOKENS")
print("=" * 60)


print(
    "Ajami PAD :",
    source_vocab.pad_idx
)

print(
    "Ajami SOS :",
    source_vocab.sos_idx
)

print(
    "Ajami EOS :",
    source_vocab.eos_idx
)

print(
    "Wolof PAD :",
    target_vocab.pad_idx
)

print(
    "Wolof SOS :",
    target_vocab.sos_idx
)

print(
    "Wolof EOS :",
    target_vocab.eos_idx
)


# ============================================================
# 4. CHARGEMENT DU CORPUS
# ============================================================

print()
print("=" * 60)
print("CHARGEMENT DU CORPUS")
print("=" * 60)


# ============================================================
# IMPORTANT
#
# Dataset reverse :
#
# SOURCE = AJAMI
# CIBLE = WOLOF LATIN
#
# Format source :
#
# Ajami + EOS
#
# Format cible :
#
# SOS + Wolof + EOS
# ============================================================

dataset = WolofAjamiReverseDataset(
    CORPUS_FILE,
    source_vocab,
    target_vocab
)


print(
    "Nombre total de paires :",
    len(dataset)
)


# ============================================================
# 5. SEPARATION TRAIN / VALIDATION / TEST
# ============================================================

print()
print("=" * 60)
print("SEPARATION TRAIN / VALIDATION / TEST")
print("=" * 60)


total_size = len(dataset)


train_size = int(
    0.80 * total_size
)


valid_size = int(
    0.10 * total_size
)


test_size = (
    total_size
    - train_size
    - valid_size
)


print(
    "Train prévu :",
    train_size
)

print(
    "Validation prévue :",
    valid_size
)

print(
    "Test prévu :",
    test_size
)


# ------------------------------------------------------------
# GENERATEUR AVEC SEED FIXE
# ------------------------------------------------------------

generator = torch.Generator().manual_seed(
    SEED
)


train_dataset, valid_dataset, test_dataset = random_split(
    dataset,
    [
        train_size,
        valid_size,
        test_size
    ],
    generator=generator
)


print()
print(
    "Paires entraînement :",
    len(train_dataset)
)

print(
    "Paires validation :",
    len(valid_dataset)
)

print(
    "Paires test :",
    len(test_dataset)
)


# ============================================================
# 6. CREATION DES DATALOADERS REVERSE
# ============================================================

print()
print("=" * 60)
print("CREATION DES DATALOADERS REVERSE")
print("=" * 60)


train_loader = create_dataloader_reverse(
    train_dataset,
    batch_size=BATCH_SIZE,
    source_pad_idx=source_vocab.pad_idx,
    target_pad_idx=target_vocab.pad_idx,
    shuffle=True
)


valid_loader = create_dataloader_reverse(
    valid_dataset,
    batch_size=BATCH_SIZE,
    source_pad_idx=source_vocab.pad_idx,
    target_pad_idx=target_vocab.pad_idx,
    shuffle=False
)


print(
    "Train batches :",
    len(train_loader)
)

print(
    "Validation batches :",
    len(valid_loader)
)


# ============================================================
# 7. CREATION DE L'ENCODEUR
# ============================================================

print()
print("=" * 60)
print("CREATION DE L'ENCODEUR")
print("=" * 60)


# SOURCE = AJAMI

encoder = EncoderGRU(
    input_dim=len(source_vocab),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE,
    pad_idx=source_vocab.pad_idx
)


print(
    "Encodeur créé."
)

print(
    "Input dimension :",
    len(source_vocab)
)

print(
    "PAD index encodeur :",
    source_vocab.pad_idx
)


# ============================================================
# 8. CREATION DE L'ATTENTION
# ============================================================

print()
print("=" * 60)
print("CREATION DE L'ATTENTION")
print("=" * 60)


attention = BahdanauAttention(
    HIDDEN_SIZE
)


print(
    "Bahdanau Attention créée."
)


# ============================================================
# 9. CREATION DU DECODEUR
# ============================================================

print()
print("=" * 60)
print("CREATION DU DECODEUR")
print("=" * 60)


# CIBLE = WOLOF LATIN

decoder = DecoderGRU(
    output_dim=len(target_vocab),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)


print(
    "Décodeur créé."
)

print(
    "Output dimension :",
    len(target_vocab)
)


# ============================================================
# 10. CREATION DU MODELE SEQ2SEQ
# ============================================================

print()
print("=" * 60)
print("CREATION DU MODELE SEQ2SEQ REVERSE")
print("=" * 60)


model = Seq2Seq(
    encoder,
    decoder,
    attention,
    DEVICE,
    src_pad_idx=source_vocab.pad_idx
).to(DEVICE)


print()
print("=" * 60)
print("MODELE AJAMI -> WOLOF CREE AVEC SUCCES")
print("=" * 60)


# ============================================================
# 11. LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=target_vocab.pad_idx
)


print(
    "PAD ignoré dans la loss :",
    target_vocab.pad_idx
)


# ============================================================
# 12. OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 13. CHEMIN DU MODELE REVERSE
# ============================================================

BEST_MODEL_PATH = (
    MODEL_DIR / "best_model_reverse.pt"
)


print()
print("=" * 60)
print("CONFIGURATION DU MODELE REVERSE")
print("=" * 60)


print(
    "Direction : AJAMI -> WOLOF LATIN"
)

print(
    "Ancien modèle Latin -> Ajami conservé :",
    MODEL_DIR / "best_model.pt"
)

print(
    "Nouveau modèle reverse :",
    BEST_MODEL_PATH
)

print(
    "Découpage : 80 % train / 10 % validation / 10 % test"
)

print(
    "Early stopping :",
    EARLY_STOPPING_PATIENCE,
    "epochs sans amélioration"
)


# ============================================================
# 14. VARIABLES DE SUIVI
# ============================================================

best_valid_loss = float("inf")

epochs_without_improvement = 0

best_epoch = 0


# ============================================================
# 15. ENTRAINEMENT
# ============================================================

print()
print("=" * 60)
print("DEBUT DE L'ENTRAINEMENT AJAMI -> WOLOF")
print("=" * 60)


for epoch in range(MAX_EPOCHS):

    epoch_start_time = time.time()

    model.train()

    epoch_loss = 0.0


    print()
    print("=" * 60)

    print(
        f"EPOCH {epoch + 1}/{MAX_EPOCHS}"
    )

    print("=" * 60)


    # ========================================================
    # TRAIN
    # ========================================================

    for batch_idx, (source, target) in enumerate(
        train_loader
    ):

        source = source.to(DEVICE)

        target = target.to(DEVICE)


        # ----------------------------------------------------
        # REMISE A ZERO DES GRADIENTS
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        output = model(
            source,
            target,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO
        )


        # ----------------------------------------------------
        # IGNORER SOS
        # ----------------------------------------------------

        output = output[:, 1:, :]


        output = output.reshape(
            -1,
            output.shape[-1]
        )


        target_loss = target[:, 1:].reshape(
            -1
        )


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = criterion(
            output,
            target_loss
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # CLIPPING DES GRADIENTS
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        # ----------------------------------------------------
        # MISE A JOUR DES POIDS
        # ----------------------------------------------------

        optimizer.step()


        epoch_loss += loss.item()


        # ----------------------------------------------------
        # AFFICHAGE
        # ----------------------------------------------------

        if (
            batch_idx == 0
            or (batch_idx + 1) % 100 == 0
        ):

            print(
                f"Batch {batch_idx + 1}/{len(train_loader)} "
                f"- Loss : {loss.item():.4f}"
            )


    # ========================================================
    # LOSS MOYENNE TRAIN
    # ========================================================

    train_loss = (
        epoch_loss
        / len(train_loader)
    )


    print()

    print(
        f"Loss entraînement : {train_loss:.4f}"
    )


    # ========================================================
    # 16. VALIDATION
    # ========================================================

    model.eval()

    valid_loss_total = 0.0


    with torch.no_grad():

        for source, target in valid_loader:

            source = source.to(DEVICE)

            target = target.to(DEVICE)


            output = model(
                source,
                target,
                teacher_forcing_ratio=0
            )


            # ------------------------------------------------
            # IGNORER SOS
            # ------------------------------------------------

            output = output[:, 1:, :]


            output = output.reshape(
                -1,
                output.shape[-1]
            )


            target_loss = target[:, 1:].reshape(
                -1
            )


            loss = criterion(
                output,
                target_loss
            )


            valid_loss_total += loss.item()


    valid_loss = (
        valid_loss_total
        / len(valid_loader)
    )


    print(
        f"Loss validation : {valid_loss:.4f}"
    )


    # ========================================================
    # 17. TEMPS
    # ========================================================

    epoch_time = (
        time.time()
        - epoch_start_time
    )


    minutes = int(
        epoch_time // 60
    )


    seconds = int(
        epoch_time % 60
    )


    print(
        f"Temps epoch : "
        f"{minutes} min {seconds} sec"
    )


    # ========================================================
    # 18. SAUVEGARDE DU MEILLEUR MODELE
    # ========================================================

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        best_epoch = epoch + 1

        epochs_without_improvement = 0


        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "train_loss":
                    train_loss,

                "valid_loss":
                    valid_loss,

                "seed":
                    SEED,

                "train_size":
                    train_size,

                "valid_size":
                    valid_size,

                "test_size":
                    test_size,

                "direction":
                    "ajami_to_wolof",

                "source_vocab_size":
                    len(source_vocab),

                "target_vocab_size":
                    len(target_vocab),

            },
            BEST_MODEL_PATH
        )


        print()
        print("=" * 60)
        print("NOUVEAU MEILLEUR MODELE REVERSE !")
        print("=" * 60)


        print(
            "Epoch :",
            best_epoch
        )


        print(
            f"Loss validation : "
            f"{best_valid_loss:.6f}"
        )


        print(
            "Modèle sauvegardé :",
            BEST_MODEL_PATH
        )


    else:

        epochs_without_improvement += 1


        print()
        print(
            "Pas d'amélioration."
        )


        print(
            "Epochs sans amélioration :",
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )


    # ========================================================
    # 19. EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print()
        print("=" * 60)
        print("EARLY STOPPING")
        print("=" * 60)


        print(
            "Aucune amélioration pendant",
            EARLY_STOPPING_PATIENCE,
            "epochs."
        )


        print(
            "Arrêt automatique de l'entraînement."
        )


        break


    print()
    print(
        f"Epoch {epoch + 1}/{MAX_EPOCHS} terminée."
    )


# ============================================================
# 20. FIN
# ============================================================

print()
print("=" * 60)
print("ENTRAINEMENT REVERSE TERMINE")
print("=" * 60)


print(
    "Direction : AJAMI -> WOLOF LATIN"
)

print(
    "Meilleure epoch :",
    best_epoch
)

print(
    "Meilleure loss validation :",
    best_valid_loss
)

print(
    "Meilleur modèle reverse :",
    BEST_MODEL_PATH
)


print()
print("=" * 60)
print("PROCHAINE ETAPE")
print("=" * 60)


print(
    "Le modèle reverse est maintenant prêt "
    "pour les tests de prédiction."
)

print(
    "Commande :"
)

print(
    "python predict_reverse.py"
)