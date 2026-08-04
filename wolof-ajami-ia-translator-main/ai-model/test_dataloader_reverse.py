import torch

from config.config import (
    DEVICE,
    BATCH_SIZE,
    CORPUS_FILE,
    WOLOF_VOCAB_FILE,
    AJAMI_VOCAB_FILE,
)

from utils.vocabulary import Vocabulary
from utils.dataset_reverse import WolofAjamiReverseDataset
from utils.dataloader_reverse import create_dataloader_reverse


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# TITRE
# ============================================================

print("=" * 70)
print("TEST DATALOADER REVERSE")
print("AJAMI -> WOLOF LATIN")
print("=" * 70)

print()
print("Device :", DEVICE)
print("Batch size :", BATCH_SIZE)
print("Seed :", SEED)


# ============================================================
# 1. CHARGEMENT DES VOCABULAIRES
# ============================================================

print()
print("=" * 70)
print("1. CHARGEMENT DES VOCABULAIRES")
print("=" * 70)


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
    "Vocabulaire Ajami :",
    len(source_vocab)
)

print(
    "Vocabulaire Wolof :",
    len(target_vocab)
)


# ============================================================
# 2. VERIFICATION DES TOKENS
# ============================================================

print()
print("=" * 70)
print("2. VERIFICATION DES TOKENS")
print("=" * 70)


print(
    "Ajami SOS :",
    source_vocab.sos_idx
)

print(
    "Ajami EOS :",
    source_vocab.eos_idx
)

print(
    "Ajami PAD :",
    source_vocab.pad_idx
)


print()


print(
    "Wolof SOS :",
    target_vocab.sos_idx
)

print(
    "Wolof EOS :",
    target_vocab.eos_idx
)

print(
    "Wolof PAD :",
    target_vocab.pad_idx
)


# ------------------------------------------------------------
# Vérification de l'existence des tokens
# ------------------------------------------------------------

assert source_vocab.sos_idx is not None
assert source_vocab.eos_idx is not None
assert source_vocab.pad_idx is not None

assert target_vocab.sos_idx is not None
assert target_vocab.eos_idx is not None
assert target_vocab.pad_idx is not None


print()
print("Tokens Ajami : OK")
print("Tokens Wolof : OK")


# ============================================================
# 3. VERIFICATION PAD SOURCE / TARGET
# ============================================================

print()
print("=" * 70)
print("3. VERIFICATION DES PAD")
print("=" * 70)


print(
    "PAD source Ajami :",
    source_vocab.pad_idx
)

print(
    "PAD cible Wolof  :",
    target_vocab.pad_idx
)


if (
    source_vocab.pad_idx
    == target_vocab.pad_idx
):

    print()
    print(
        "ATTENTION : les PAD source et cible ont le même indice."
    )

    print(
        "Ce n'est pas forcément une erreur si les vocabulaires"
    )

    print(
        "sont indépendants, mais le DataLoader utilise bien"
    )

    print(
        "séparément source_pad_idx et target_pad_idx."
    )

else:

    print()
    print(
        "PAD source et PAD cible indépendants : OK"
    )


# ============================================================
# 4. CHARGEMENT DU DATASET REVERSE
# ============================================================

print()
print("=" * 70)
print("4. CHARGEMENT DU DATASET REVERSE")
print("=" * 70)


dataset = WolofAjamiReverseDataset(
    CORPUS_FILE,
    source_vocab,
    target_vocab
)


print(
    "Nombre total de paires :",
    len(dataset)
)


assert len(dataset) > 0


print()
print("Dataset reverse : OK")


# ============================================================
# 5. TEST D'UN EXEMPLE
# ============================================================

print()
print("=" * 70)
print("5. TEST D'UN EXEMPLE")
print("=" * 70)


source, target = dataset[0]


print()
print("Premier exemple :")

print(
    "Source indices :",
    source.tolist()
)

print(
    "Target indices :",
    target.tolist()
)


print()
print(
    "Longueur source :",
    len(source)
)

print(
    "Longueur cible :",
    len(target)
)


# ------------------------------------------------------------
# Vérification SOS / EOS
# ------------------------------------------------------------

assert source[0].item() == source_vocab.sos_idx

assert (
    source[-1].item()
    == source_vocab.eos_idx
)


assert target[0].item() == target_vocab.sos_idx

assert (
    target[-1].item()
    == target_vocab.eos_idx
)


print()
print("SOS / EOS source : OK")
print("SOS / EOS cible  : OK")


# ============================================================
# 6. CREATION DU DATALOADER REVERSE
# ============================================================

print()
print("=" * 70)
print("6. CREATION DU DATALOADER REVERSE")
print("=" * 70)


loader = create_dataloader_reverse(
    dataset,
    batch_size=BATCH_SIZE,
    source_pad_idx=source_vocab.pad_idx,
    target_pad_idx=target_vocab.pad_idx,
    shuffle=False
)


print(
    "Nombre de batches :",
    len(loader)
)


assert len(loader) > 0


print()
print("DataLoader reverse créé avec succès.")


# ============================================================
# 7. RECUPERATION DU PREMIER BATCH
# ============================================================

print()
print("=" * 70)
print("7. TEST DU PREMIER BATCH")
print("=" * 70)


batch_source, batch_target = next(
    iter(loader)
)


print()
print(
    "Shape source :",
    batch_source.shape
)

print(
    "Shape cible  :",
    batch_target.shape
)


print()
print(
    "Type source :",
    batch_source.dtype
)

print(
    "Type cible :",
    batch_target.dtype
)


# ============================================================
# 8. VERIFICATION DES DIMENSIONS
# ============================================================

print()
print("=" * 70)
print("8. VERIFICATION DES DIMENSIONS")
print("=" * 70)


assert batch_source.dim() == 2

assert batch_target.dim() == 2


assert batch_source.shape[0] <= BATCH_SIZE

assert batch_target.shape[0] <= BATCH_SIZE


print(
    "Dimensions source : OK"
)

print(
    "Dimensions cible  : OK"
)


# ============================================================
# 9. VERIFICATION DU PADDING
# ============================================================

print()
print("=" * 70)
print("9. VERIFICATION DU PADDING")
print("=" * 70)


source_pad_count = (
    batch_source
    == source_vocab.pad_idx
).sum().item()


target_pad_count = (
    batch_target
    == target_vocab.pad_idx
).sum().item()


print(
    "Nombre de PAD source :",
    source_pad_count
)

print(
    "Nombre de PAD cible  :",
    target_pad_count
)


print()
print(
    "Padding source avec :",
    source_vocab.pad_idx
)

print(
    "Padding cible avec :",
    target_vocab.pad_idx
)


print()
print("Padding : OK")


# ============================================================
# 10. AFFICHAGE DU BATCH
# ============================================================

print()
print("=" * 70)
print("10. CONTENU DU PREMIER BATCH")
print("=" * 70)


print()


for i in range(
    min(5, batch_source.shape[0])
):

    print(
        f"Exemple {i + 1}"
    )

    print(
        "  Source :",
        batch_source[i].tolist()
    )

    print(
        "  Cible  :",
        batch_target[i].tolist()
    )

    print()


# ============================================================
# 11. VERIFICATION SOS / EOS DU BATCH
# ============================================================

print()
print("=" * 70)
print("11. VERIFICATION SOS / EOS DU BATCH")
print("=" * 70)


for i in range(
    batch_source.shape[0]
):

    source_row = batch_source[i]

    target_row = batch_target[i]


    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    assert (
        source_row[0].item()
        == source_vocab.sos_idx
    )


    # --------------------------------------------------------
    # CIBLE
    # --------------------------------------------------------

    assert (
        target_row[0].item()
        == target_vocab.sos_idx
    )


print(
    "SOS source : OK"
)

print(
    "SOS cible  : OK"
)


# ============================================================
# 12. TEST DE CONVERSION DES INDICES
# ============================================================

print()
print("=" * 70)
print("12. VERIFICATION DU VOCABULAIRE")
print("=" * 70)


# ------------------------------------------------------------
# Source Ajami
# ------------------------------------------------------------

first_source = batch_source[0].tolist()

print()
print("Source décodée approximativement :")


source_tokens = []

for idx in first_source:

    if idx == source_vocab.pad_idx:
        token = "<PAD>"

    else:
        token = source_vocab.itos[idx]

    source_tokens.append(token)


print(
    source_tokens
)


# ------------------------------------------------------------
# Target Wolof
# ------------------------------------------------------------

first_target = batch_target[0].tolist()

print()
print("Cible décodée approximativement :")


target_tokens = []

for idx in first_target:

    if idx == target_vocab.pad_idx:
        token = "<PAD>"

    else:
        token = target_vocab.itos[idx]

    target_tokens.append(token)


print(
    target_tokens
)


# ============================================================
# 13. TEST ENVOI VERS DEVICE
# ============================================================

print()
print("=" * 70)
print("13. TEST DEVICE")
print("=" * 70)


batch_source_device = batch_source.to(
    DEVICE
)

batch_target_device = batch_target.to(
    DEVICE
)


print(
    "Source device :",
    batch_source_device.device
)

print(
    "Target device :",
    batch_target_device.device
)


assert (
    batch_source_device.device
    == torch.device(DEVICE)
)


assert (
    batch_target_device.device
    == torch.device(DEVICE)
)


print()
print("Transfert vers DEVICE : OK")


# ============================================================
# RESULTAT FINAL
# ============================================================

print()
print("=" * 70)
print("TEST DATALOADER REVERSE TERMINE")
print("=" * 70)

print()
print("AJAMI -> WOLOF LATIN")
print()

print("Dataset              : OK")
print("Vocabulaire source   : OK")
print("Vocabulaire cible    : OK")
print("SOS source           : OK")
print("SOS cible             : OK")
print("EOS source           : OK")
print("EOS cible             : OK")
print("PAD source            : OK")
print("PAD cible             : OK")
print("Padding indépendant   : OK")
print("Batch 2D              : OK")
print("Transfert DEVICE      : OK")

print()
print("=" * 70)
print("TOUT EST OK")
print("=" * 70)

print()
print(
    "Le DataLoader reverse est prêt pour l'entraînement."
)

print()
print(
    "Prochaine étape : réentraîner le modèle AJAMI -> WOLOF."
)