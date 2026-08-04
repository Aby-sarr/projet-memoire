import torch
import torch.nn as nn


class EncoderGRU(nn.Module):
    """
    Encodeur GRU pour le modèle Seq2Seq.
    """

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        num_layers=1
    ):
        super(EncoderGRU, self).__init__()

        # Transformation des indices en vecteurs
        self.embedding = nn.Embedding(
            input_dim,
            embedding_dim
        )

        # Réseau GRU
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )


    def forward(self, src):

        # src :
        # [batch_size, longueur_sequence]

        embedded = self.embedding(src)

        # embedded :
        # [batch_size, longueur_sequence, embedding_dim]

        outputs, hidden = self.gru(
            embedded
        )

        # outputs :
        # tous les états cachés h1,h2,...hn
        #
        # hidden :
        # dernier état caché

        return outputs, hidden