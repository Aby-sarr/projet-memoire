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

from utils.dataloader import (
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

EARLY_STOPPING_PATIENCE = 5


# ============================================================
# SEED
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# TITRE
# ============================================================

print()
print("=" * 70)
print("ENTRAINEMENT MODELE REVERSE")
print("AJAMI -> WOLOF LATIN")
print("=" * 70)

print()

print("Device :", DEVICE)

print("Batch size :", BATCH_SIZE)

print("Embedding :", EMBEDDING)

print("Hidden size :", HIDDEN_SIZE)

print("Nombre maximum d'epochs :", MAX_EPOCHS)

print("Learning rate :", LEARNING_RATE)

print(
    "Teacher forcing :",
    TEACHER_FORCING_RATIO
)

print(
    "Early stopping patience :",
    EARLY_STOPPING_PATIENCE
)

print("Seed :", SEED)


# ============================================================
# VOCABULAIRES
# ============================================================

print()
print("=" * 70)
print("CHARGEMENT DES VOCABULAIRES")
print("=" * 70)


# SOURCE = AJAMI

source_vocab = Vocabulary(
    AJAMI_VOCAB_FILE
)


# TARGET = WOLOF

target_vocab = Vocabulary(
    WOLOF_VOCAB_FILE
)


print(
    "Vocabulaire Ajami :",
    len(source_vocab)
)


print(
    "Vocabulaire Wolof :",
    len(target_vocab)
)


print()

print(
    "PAD Ajami :",
    source_vocab.pad_idx
)

print(
    "SOS Ajami :",
    source_vocab.sos_idx
)

print(
    "EOS Ajami :",
    source_vocab.eos_idx
)

print(
    "UNK Ajami :",
    source_vocab.unk_idx
)


print()

print(
    "PAD Wolof :",
    target_vocab.pad_idx
)

print(
    "SOS Wolof :",
    target_vocab.sos_idx
)

print(
    "EOS Wolof :",
    target_vocab.eos_idx
)

print(
    "UNK Wolof :",
    target_vocab.unk_idx
)


# ============================================================
# VERIFICATION VOCABULAIRE
# ============================================================

print()
print("=" * 70)
print("VERIFICATION DES TOKENS")
print("=" * 70)


required_wolof = [
    "C",
    "c",
    "a",
    "n",
    "j",
    "g",
    "ñ",
]


for char in required_wolof:

    if char in target_vocab.stoi:

        print(
            f"Wolof '{char}' ->",
            target_vocab.stoi[char]
        )

    else:

        print(
            f"ATTENTION : '{char}' absent du vocabulaire"
        )


# ============================================================
# DATASET
# ============================================================

print()
print("=" * 70)
print("CHARGEMENT DATASET REVERSE")
print("=" * 70)


dataset = WolofAjamiReverseDataset(
    CORPUS_FILE,
    source_vocab,
    target_vocab
)


print()

print(
    "Nombre total de paires :",
    len(dataset)
)


# ============================================================
# VERIFICATION DIRECTE DU DATASET
# ============================================================

print()
print("=" * 70)
print("VERIFICATION DES PREMIERES PAIRES")
print("=" * 70)


for i in range(min(5, len(dataset))):

    source, target = dataset[i]


    print()
    print(
        f"Paire {i + 1}"
    )


    print(
        "Source indices :",
        source.tolist()
    )


    print(
        "Cible indices :",
        target.tolist()
    )


    print(
        "Source texte :",
        source_vocab.decode(
            source.tolist()
        )
    )


    print(
        "Cible texte :",
        target_vocab.decode(
            target.tolist()
        )
    )


# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

print()
print("=" * 70)
print("SEPARATION TRAIN / VALIDATION / TEST")
print("=" * 70)


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
    "Train :",
    train_size
)

print(
    "Validation :",
    valid_size
)

print(
    "Test :",
    test_size
)


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
    "Train réel :",
    len(train_dataset)
)

print(
    "Validation réelle :",
    len(valid_dataset)
)

print(
    "Test réel :",
    len(test_dataset)
)


# ============================================================
# DATALOADERS
# ============================================================

print()
print("=" * 70)
print("CREATION DES DATALOADERS")
print("=" * 70)


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
# VERIFICATION D'UN BATCH
# ============================================================

print()
print("=" * 70)
print("VERIFICATION D'UN BATCH")
print("=" * 70)


sample_source, sample_target = next(
    iter(train_loader)
)


print(
    "Shape source :",
    tuple(sample_source.shape)
)


print(
    "Shape cible :",
    tuple(sample_target.shape)
)


print(
    "Premier source :",
    sample_source[0].tolist()
)


print(
    "Première cible :",
    sample_target[0].tolist()
)


# ============================================================
# MODELE
# ============================================================

print()
print("=" * 70)
print("CONSTRUCTION DU MODELE REVERSE")
print("=" * 70)


# ------------------------------------------------------------
# ENCODEUR
# ------------------------------------------------------------

encoder = EncoderGRU(
    input_dim=len(source_vocab),
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
# ------------------------------------------------------------

decoder = DecoderGRU(
    output_dim=len(target_vocab),
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


print()

print(
    "Modèle AJAMI -> WOLOF créé."
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=target_vocab.pad_idx
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# CHEMIN MODELE
# ============================================================

BEST_MODEL_PATH = (
    MODEL_DIR / "best_model_reverse.pt"
)


print()
print("=" * 70)
print("CONFIGURATION SAUVEGARDE")
print("=" * 70)


print(
    "Ancien modèle Latin -> Ajami :"
)


print(
    MODEL_DIR / "best_model.pt"
)


print()

print(
    "Nouveau modèle Ajami -> Wolof :"
)


print(
    BEST_MODEL_PATH
)


# ============================================================
# VARIABLES
# ============================================================

best_valid_loss = float("inf")

epochs_without_improvement = 0

best_epoch = 0


# ============================================================
# ENTRAINEMENT
# ============================================================

print()
print("=" * 70)
print("DEBUT ENTRAINEMENT")
print("AJAMI -> WOLOF LATIN")
print("=" * 70)


for epoch in range(MAX_EPOCHS):


    start_time = time.time()


    # ========================================================
    # TRAIN MODE
    # ========================================================

    model.train()


    epoch_loss = 0.0


    print()
    print("=" * 70)

    print(
        f"EPOCH {epoch + 1}/{MAX_EPOCHS}"
    )

    print("=" * 70)


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for batch_idx, (
        source,
        target
    ) in enumerate(train_loader):


        source = source.to(
            DEVICE
        )


        target = target.to(
            DEVICE
        )


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
        # OUTPUT
        # ----------------------------------------------------

        output_dim = output.shape[-1]


        # output :
        #
        # [batch, target_len, vocab]
        #
        # On ignore la position 0
        # correspondant à SOS.


        output = output[:, 1:, :]


        target_loss = target[:, 1:]


        output = output.reshape(
            -1,
            output_dim
        )


        target_loss = target_loss.reshape(
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
        # GRADIENT CLIPPING
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        # ----------------------------------------------------
        # OPTIMISATION
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
                f"Batch "
                f"{batch_idx + 1}/"
                f"{len(train_loader)} "
                f"- Loss : "
                f"{loss.item():.6f}"
            )


    # ========================================================
    # TRAIN LOSS
    # ========================================================

    train_loss = (
        epoch_loss
        / len(train_loader)
    )


    print()

    print(
        f"Loss entraînement : "
        f"{train_loss:.6f}"
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    valid_loss_total = 0.0


    with torch.no_grad():


        for source, target in valid_loader:


            source = source.to(
                DEVICE
            )


            target = target.to(
                DEVICE
            )


            output = model(
                source,
                target,
                teacher_forcing_ratio=0
            )


            output_dim = output.shape[-1]


            output = output[:, 1:, :]


            target_loss = target[:, 1:]


            output = output.reshape(
                -1,
                output_dim
            )


            target_loss = target_loss.reshape(
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
        f"Loss validation : "
        f"{valid_loss:.6f}"
    )


    # ========================================================
    # TEMPS
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )


    minutes = int(
        elapsed // 60
    )


    seconds = int(
        elapsed % 60
    )


    print(
        f"Temps epoch : "
        f"{minutes} min {seconds} sec"
    )


    # ========================================================
    # SAUVEGARDE
    # ========================================================

    if valid_loss < best_valid_loss:


        best_valid_loss = valid_loss

        best_epoch = epoch + 1

        epochs_without_improvement = 0


        torch.save(
            {
                "epoch":
                    epoch + 1,

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

                "embedding_dim":
                    EMBEDDING,

                "hidden_size":
                    HIDDEN_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "teacher_forcing_ratio":
                    TEACHER_FORCING_RATIO,
            },
            BEST_MODEL_PATH
        )


        print()
        print("=" * 70)
        print("NOUVEAU MEILLEUR MODELE REVERSE")
        print("=" * 70)


        print(
            "Epoch :",
            best_epoch
        )


        print(
            "Validation loss :",
            best_valid_loss
        )


        print(
            "Fichier :",
            BEST_MODEL_PATH
        )


    else:


        epochs_without_improvement += 1


        print()

        print(
            "Pas d'amélioration."
        )


        print(
            "Patience :",
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):


        print()
        print("=" * 70)
        print("EARLY STOPPING")
        print("=" * 70)


        print(
            "Arrêt après",
            EARLY_STOPPING_PATIENCE,
            "epochs sans amélioration."
        )


        break


# ============================================================
# FIN
# ============================================================

print()
print("=" * 70)
print("ENTRAINEMENT REVERSE TERMINE")
print("=" * 70)


print(
    "Direction : AJAMI -> WOLOF LATIN"
)


print(
    "Meilleure epoch :",
    best_epoch
)


print(
    "Meilleure validation loss :",
    best_valid_loss
)


print(
    "Modèle :",
    BEST_MODEL_PATH
)


print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)


print(
    "Ne lancez PAS encore l'application web."
)


print(
    "Il faut d'abord vérifier le modèle avec"
)


print(
    "le diagnostic teacher forcing."
)


print()
print(
    "Ensuite seulement :"
)


print(
    "python evaluate_test_reverse.py"
)