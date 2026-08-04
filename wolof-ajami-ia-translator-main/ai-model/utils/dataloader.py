import torch

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader


def collate_fn(
    batch,
    source_pad_idx,
    target_pad_idx
):
    """
    Prépare un batch pour :

        AJAMI -> WOLOF

    avec un PAD indépendant pour la source
    et la cible.
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



def create_dataloader_reverse(
    dataset,
    batch_size,
    source_pad_idx,
    target_pad_idx,
    shuffle=True
):
    """
    DataLoader spécifique au modèle reverse.

    AJAMI -> WOLOF LATIN
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch:
            collate_fn(
                batch,
                source_pad_idx,
                target_pad_idx
            )
    )


    return loader