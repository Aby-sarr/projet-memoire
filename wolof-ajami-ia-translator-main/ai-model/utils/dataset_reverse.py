import pandas as pd
import torch
from torch.utils.data import Dataset


class WolofAjamiReverseDataset(Dataset):
    """
    Dataset pour la translittération :

        AJAMI -> WOLOF LATIN

    Source :
        colonne "ajami"

    Cible :
        colonne "Wolof"
    """

    def __init__(
        self,
        csv_file,
        source_vocab,
        target_vocab
    ):

        self.data = pd.read_csv(
            csv_file,
            encoding="utf-8"
        )

        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        # Vérification des colonnes
        required_columns = ["ajami", "Wolof"]

        for column in required_columns:

            if column not in self.data.columns:

                raise ValueError(
                    f"Colonne '{column}' absente du corpus. "
                    f"Colonnes disponibles : {list(self.data.columns)}"
                )

        print()
        print("=" * 60)
        print("DATASET REVERSE")
        print("=" * 60)

        print(
            "Source : AJAMI"
        )

        print(
            "Cible  : WOLOF LATIN"
        )

        print(
            "Nombre de paires :",
            len(self.data)
        )


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        # ====================================================
        # LIGNE DU CORPUS
        # ====================================================

        row = self.data.iloc[index]


        # ====================================================
        # SOURCE AJAMI
        # ====================================================

        source_text = str(
            row["ajami"]
        ).strip()


        # ====================================================
        # CIBLE WOLOF
        # ====================================================

        target_text = str(
            row["Wolof"]
        ).strip()


        # ====================================================
        # ENCODAGE SOURCE
        # ====================================================

        source_ids = self.source_vocab.encode(
            source_text
        )


        # ====================================================
        # ENCODAGE CIBLE
        # ====================================================

        target_ids = self.target_vocab.encode(
            target_text
        )


        # ====================================================
        # AJOUT SOS / EOS
        # ====================================================

        source_ids = (
            [self.source_vocab.sos_idx]
            + source_ids
            + [self.source_vocab.eos_idx]
        )


        target_ids = (
            [self.target_vocab.sos_idx]
            + target_ids
            + [self.target_vocab.eos_idx]
        )


        # ====================================================
        # TENSEURS
        # ====================================================

        source_tensor = torch.tensor(
            source_ids,
            dtype=torch.long
        )


        target_tensor = torch.tensor(
            target_ids,
            dtype=torch.long
        )


        return (
            source_tensor,
            target_tensor
        )