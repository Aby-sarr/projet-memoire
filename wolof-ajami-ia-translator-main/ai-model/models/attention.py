import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    """
    Mécanisme d'attention de Bahdanau.
    """

    def __init__(self, hidden_dim):

        super().__init__()

        self.W = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.V = nn.Linear(
            hidden_dim,
            1,
            bias=False
        )


    def forward(
        self,
        hidden,
        encoder_outputs
    ):

        # hidden :
        # [1, batch, hidden_dim]

        # encoder_outputs :
        # [batch, source_length, hidden_dim]


        hidden = hidden.permute(1,0,2)


        src_len = encoder_outputs.size(1)


        hidden = hidden.repeat(
            1,
            src_len,
            1
        )


        energy = torch.tanh(
            self.W(
                torch.cat(
                    (
                        hidden,
                        encoder_outputs
                    ),
                    dim=2
                )
            )
        )


        attention = self.V(
            energy
        ).squeeze(2)


        weights = torch.softmax(
            attention,
            dim=1
        )


        context = torch.bmm(
            weights.unsqueeze(1),
            encoder_outputs
        )


        return context, weights