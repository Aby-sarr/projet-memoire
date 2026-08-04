import torch

from models.attention import BahdanauAttention


# paramètres
HIDDEN_SIZE = 512
BATCH_SIZE = 4
SOURCE_LENGTH = 10


# création de l'attention
attention = BahdanauAttention(
    HIDDEN_SIZE
)


# simulation des sorties de l'encodeur
encoder_outputs = torch.randn(
    BATCH_SIZE,
    SOURCE_LENGTH,
    HIDDEN_SIZE
)


# simulation de l'état caché du décodeur
hidden = torch.randn(
    1,
    BATCH_SIZE,
    HIDDEN_SIZE
)


# calcul attention
context, weights = attention(
    hidden,
    encoder_outputs
)


print("Vecteur contexte :")
print(context.shape)


print("\nPoids attention :")
print(weights.shape)


print("\nSomme des poids :")
print(weights.sum(dim=1))