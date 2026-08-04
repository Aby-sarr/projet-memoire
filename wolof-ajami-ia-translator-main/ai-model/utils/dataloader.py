import torch

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader


# ============================================================
# COLLATE FUNCTION
# MODELE NORMAL
# WOLOF LATIN -> AJAMI
# ============================================================

def collate_fn(
    batch,
    pad_idx
):
    """
    Prépare un batch pour le modèle normal :

        WOLOF LATIN -> WOLOF AJAMI

    Le padding est effectué avec le PAD
    du vocabulaire source.
    """

    sources = []
    targets = []

    for source, target in batch:

        sources.append(source)
        targets.append(target)

    # ========================================================
    # PADDING SOURCE
    # ========================================================

    sources = pad_sequence(
        sources,
        batch_first=True,
        padding_value=pad_idx
    )

    # ========================================================
    # PADDING CIBLE
    # ========================================================

    targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=pad_idx
    )

    return (
        sources,
        targets
    )


# ============================================================
# DATALOADER MODELE NORMAL
# WOLOF LATIN -> AJAMI
# ============================================================

def create_dataloader(
    dataset,
    batch_size,
    pad_idx,
    shuffle=True
):
    """
    DataLoader du modèle principal.

    Direction :

        WOLOF LATIN -> WOLOF AJAMI
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch:
            collate_fn(
                batch,
                pad_idx
            )
    )

    return loader