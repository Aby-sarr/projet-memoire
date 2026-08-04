import torch
import torch.nn as nn


class DecoderGRU(nn.Module):
    """
    Décodeur GRU avec attention Bahdanau.
    """

    def __init__(
        self,
        output_dim,
        embedding_dim,
        hidden_dim
    ):
        super().__init__()


        self.output_dim = output_dim


        self.embedding = nn.Embedding(
            output_dim,
            embedding_dim
        )


        self.gru = nn.GRU(
            embedding_dim + hidden_dim,
            hidden_dim,
            batch_first=True
        )


        self.fc_out = nn.Linear(
            hidden_dim * 2,
            output_dim
        )


    def forward(
        self,
        input,
        hidden,
        context
    ):

        # input :
        # [batch_size]


        input = input.unsqueeze(1)


        embedded = self.embedding(
            input
        )


        # concaténation embedding + contexte

        context = context
        

        rnn_input = torch.cat(
            (
                embedded,
                context
            ),
            dim=2
        )


        output, hidden = self.gru(
            rnn_input,
            hidden
        )


        prediction = self.fc_out(
            torch.cat(
                (
                    output.squeeze(1),
                    context.squeeze(1)
                ),
                dim=1
            )
        )


        return prediction, hidden