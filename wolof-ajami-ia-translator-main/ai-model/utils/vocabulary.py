import json


class Vocabulary:
    """
    Gestionnaire de vocabulaire pour la translittération
    Wolof Latin ↔ Wolof Ajami.
    """

    def __init__(self, vocab_path):

        with open(
            vocab_path,
            "r",
            encoding="utf-8"
        ) as f:

            vocab = json.load(f)

        self.stoi = vocab["stoi"]
        self.itos = vocab["itos"]

        # ----------------------------------------------------
        # Conversion des clés de itos en entiers
        # ----------------------------------------------------

        if isinstance(self.itos, dict):

            self.itos = {
                int(index): caractere
                for index, caractere in self.itos.items()
            }

        # ----------------------------------------------------
        # Tokens spéciaux
        # ----------------------------------------------------

        self.pad_idx = self.stoi["<PAD>"]
        self.sos_idx = self.stoi["<SOS>"]
        self.eos_idx = self.stoi["<EOS>"]
        self.unk_idx = self.stoi["<UNK>"]

    # --------------------------------------------------------
    # Taille du vocabulaire
    # --------------------------------------------------------

    def __len__(self):

        return len(self.stoi)

    # --------------------------------------------------------
    # Encodage
    # --------------------------------------------------------

    def encode(self, text):

        tokens = []

        for char in str(text):

            tokens.append(
                self.stoi.get(
                    char,
                    self.unk_idx
                )
            )

        return tokens

    # --------------------------------------------------------
    # Décodage
    # --------------------------------------------------------

    def decode(self, indices):

        chars = []

        for idx in indices:

            idx = int(idx)

            if idx == self.eos_idx:
                break

            if idx not in self.itos:
                continue

            chars.append(
                self.itos[idx]
            )

        return "".join(chars)