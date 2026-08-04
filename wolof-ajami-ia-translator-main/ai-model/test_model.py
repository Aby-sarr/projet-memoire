import torch

from config.config import (
    WOLOF_VOCAB_FILE,
    AJAMI_VOCAB_FILE,
    MODEL_DIR,
    EMBEDDING,
    HIDDEN_SIZE,
    DEVICE,
)

from utils.vocabulary import Vocabulary
from utils.dataset import WolofAjamiDataset

from models.encoder import EncoderGRU
from models.attention import BahdanauAttention
from models.decoder import DecoderGRU
from models.seq2seq import Seq2Seq


print("=" * 50)
print("Chargement des vocabulaires...")

wolof_vocab = Vocabulary(WOLOF_VOCAB_FILE)
ajami_vocab = Vocabulary(AJAMI_VOCAB_FILE)

print("Vocabulaire Wolof :", len(wolof_vocab))
print("Vocabulaire Ajami :", len(ajami_vocab))


print("=" * 50)
print("Construction du modèle...")

encoder = EncoderGRU(
    input_dim=len(wolof_vocab),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)

attention = BahdanauAttention(
    HIDDEN_SIZE
)

decoder = DecoderGRU(
    output_dim=len(ajami_vocab),
    embedding_dim=EMBEDDING,
    hidden_dim=HIDDEN_SIZE
)

model = Seq2Seq(
    encoder,
    decoder,
    attention,
    DEVICE
).to(DEVICE)


print("=" * 50)
print("Chargement du modèle sauvegardé...")

MODEL_PATH = MODEL_DIR / "best_model.pt"

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Modèle chargé avec succès !")
print("Epoch sauvegardée :", checkpoint["epoch"])
print("Loss entraînement :", checkpoint["train_loss"])
print("Loss validation :", checkpoint["valid_loss"])


print("=" * 50)
print("Test de chargement terminé.")
print("Le modèle est prêt.")