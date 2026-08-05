import torch
import torch.nn as nn


class EncoderGRU(nn.Module):
    """
    Encodeur GRU.

    Transforme une séquence de tokens source
    en représentations contextuelles.
    """

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        pad_idx=0
    ):

        super().__init__()

        self.hidden_dim = hidden_dim

        self.pad_idx = pad_idx


        # ====================================================
        # EMBEDDING
        # ====================================================

        self.embedding = nn.Embedding(
            input_dim,
            embedding_dim,
            padding_idx=pad_idx
        )


        # ====================================================
        # GRU
        # ====================================================

        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )


    def forward(
        self,
        src
    ):

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        embedded = self.embedding(
            src
        )


        # ----------------------------------------------------
        # GRU
        # ----------------------------------------------------

        outputs, hidden = self.rnn(
            embedded
        )


        return outputs, hidden