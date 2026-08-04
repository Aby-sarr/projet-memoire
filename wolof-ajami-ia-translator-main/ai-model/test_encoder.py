import torch

from models.encoder import EncoderGRU
from utils.vocabulary import Vocabulary


vocab = Vocabulary(
    "data/vocab_wolof.json"
)


encoder = EncoderGRU(
    input_dim=len(vocab),
    embedding_dim=256,
    hidden_dim=512
)


# Exemple batch fictif
src = torch.tensor([
    [1,54,35,58,35,57,35,46,2]
])


outputs, hidden = encoder(src)


print("Sortie encodeur :")
print(outputs.shape)


print("\nEtat caché :")
print(hidden.shape)