import torch

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader


# ============================================================
# COLLATE FUNCTION REVERSE
# AJAMI -> WOLOF LATIN
# ============================================================

def collate_fn_reverse(
    batch,
    source_pad_idx,
    target_pad_idx
):
    """
    Prépare un batch pour le modèle reverse :

        WOLOF AJAMI -> WOLOF LATIN

    La source et la cible possèdent des vocabulaires
    différents.

    Source :
        vocabulaire Ajami

    Cible :
        vocabulaire Wolof Latin
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
        padding_value=source_pad_idx
    )

    # ========================================================
    # PADDING CIBLE
    # ========================================================

    targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=target_pad_idx
    )

    return (
        sources,
        targets
    )


# ============================================================
# DATALOADER REVERSE
# AJAMI -> WOLOF LATIN
# ============================================================

def create_dataloader_reverse(
    dataset,
    batch_size,
    source_pad_idx,
    target_pad_idx,
    shuffle=True
):
    """
    DataLoader spécifique au modèle reverse.

    Direction :

        WOLOF AJAMI -> WOLOF LATIN
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch:
            collate_fn_reverse(
                batch,
                source_pad_idx,
                target_pad_idx
            )
    )

    return loader