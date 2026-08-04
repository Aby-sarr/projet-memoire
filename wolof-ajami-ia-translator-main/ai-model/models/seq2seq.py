import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    """
    Modèle Seq2Seq complet :
    Encodeur GRU + Attention Bahdanau + Décodeur GRU
    """

    def __init__(
        self,
        encoder,
        decoder,
        attention,
        device
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.device = device


    def forward(
        self,
        src,
        trg,
        teacher_forcing_ratio=0.5
    ):

        batch_size = src.shape[0]

        trg_length = trg.shape[1]

        trg_vocab_size = self.decoder.output_dim


        outputs = torch.zeros(
            batch_size,
            trg_length,
            trg_vocab_size
        ).to(self.device)


        # Passage dans l'encodeur

        encoder_outputs, hidden = self.encoder(src)


        # Premier token du décodeur = SOS

        input = trg[:,0]


        for t in range(1, trg_length):

            # Calcul attention

            context, attention_weights = self.attention(
                hidden,
                encoder_outputs
            )


            # Décodeur

            output, hidden = self.decoder(
                input,
                hidden,
                context
            )


            outputs[:,t,:] = output


            # Teacher forcing

            best_guess = output.argmax(1)


            teacher_force = (
                torch.rand(1).item()
                < teacher_forcing_ratio
            )


            input = (
                trg[:,t]
                if teacher_force
                else best_guess
            )


        return outputs