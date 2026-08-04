import pandas as pd
import torch
from torch.utils.data import Dataset


class WolofAjamiDataset(Dataset):
    """
    Dataset PyTorch pour la translittération
    Wolof Latin ↔ Wolof Ajami.
    """

    def __init__(
        self,
        csv_file,
        source_vocab,
        target_vocab
    ):

        self.data = pd.read_csv(
            csv_file
        )

        self.source_vocab = source_vocab
        self.target_vocab = target_vocab


    def __len__(self):
        """
        Nombre total de paires.
        """
        return len(self.data)


    def __getitem__(self, index):

        source_text = str(
            self.data.iloc[index]["Wolof"]
        )

        target_text = str(
            self.data.iloc[index]["ajami"]
        )


        # Encodage caractère par caractère
        source_ids = self.source_vocab.encode(
            source_text
        )

        target_ids = self.target_vocab.encode(
            target_text
        )


        # Ajout des tokens début et fin
        source_ids = [
            self.source_vocab.sos_idx
        ] + source_ids + [
            self.source_vocab.eos_idx
        ]


        target_ids = [
            self.target_vocab.sos_idx
        ] + target_ids + [
            self.target_vocab.eos_idx
        ]


        return (
            torch.tensor(
                source_ids,
                dtype=torch.long
            ),
            torch.tensor(
                target_ids,
                dtype=torch.long
            )
        )