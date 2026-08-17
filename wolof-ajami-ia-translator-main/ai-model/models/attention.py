import torch
import torch.nn as nn
class BahdanauAttention(nn.Module):

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
        encoder_outputs,
        mask=None
    ):

        hidden = hidden.permute(1, 0, 2)

        src_len = encoder_outputs.size(1)

        hidden = hidden.repeat(1, src_len, 1)

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

        # [batch, source_length]
        attention = self.V(
            energy
        ).squeeze(2)

        # --------------------------------------------------
        # MASQUE DES <PAD>
        # --------------------------------------------------

        if mask is not None:

            # Les positions PAD ne doivent recevoir
            # aucune probabilité d'attention.
            attention = attention.masked_fill(
                mask == 0,
                -1e10
            )

        # Softmax uniquement sur les positions valides
        weights = torch.softmax(
            attention,
            dim=1
        )

        # --------------------------------------------------
        # VECTEUR CONTEXTE
        # --------------------------------------------------

        # [batch, 1, hidden_dim]
        context = torch.bmm(
            weights.unsqueeze(1),
            encoder_outputs
        )

        return context, weights