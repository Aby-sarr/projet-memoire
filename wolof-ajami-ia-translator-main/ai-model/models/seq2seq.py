import torch
import torch.nn as nn
class Seq2Seq(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        attention,
        device,
        src_pad_idx=0
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.device = device

        # Indice du PAD dans le vocabulaire source
        self.src_pad_idx = src_pad_idx

    def create_src_mask(self, src):
        """
        Crée le masque de la séquence source.

        1 = vraie position
        0 = PAD

        src :
        [batch, source_length]

        mask :
        [batch, source_length]
        """

        return (
            src != self.src_pad_idx
        )

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
            trg_vocab_size,
            device=self.device
        )

        # ----------------------------------------------------
        # MASQUE SOURCE
        # ----------------------------------------------------

        src_mask = self.create_src_mask(
            src
        )

        # ----------------------------------------------------
        # ENCODEUR
        # ----------------------------------------------------

        encoder_outputs, hidden = self.encoder(
            src
        )

        # ----------------------------------------------------
        # PREMIER TOKEN = SOS
        # ----------------------------------------------------

        input = trg[:, 0]

        # ----------------------------------------------------
        # DECODAGE
        # ----------------------------------------------------

        for t in range(
            1,
            trg_length
        ):

            # ------------------------------------------------
            # ATTENTION AVEC MASQUE
            # ------------------------------------------------

            context, attention_weights = self.attention(
                hidden,
                encoder_outputs,
                mask=src_mask
            )

            # ------------------------------------------------
            # DECODEUR
            # ------------------------------------------------

            output, hidden = self.decoder(
                input,
                hidden,
                context
            )

            outputs[:, t, :] = output

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            best_guess = output.argmax(
                dim=1
            )

            # ------------------------------------------------
            # TEACHER FORCING
            # ------------------------------------------------

            teacher_force = (
                torch.rand(1).item()
                < teacher_forcing_ratio
            )

            input = (
                trg[:, t]
                if teacher_force
                else best_guess
            )

        return outputs