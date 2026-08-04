import torch

from models.decoder import DecoderGRU


# Paramètres
OUTPUT_DIM = 51   # taille du vocabulaire Ajami
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 4


# Création du décodeur
decoder = DecoderGRU(
    output_dim=OUTPUT_DIM,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_SIZE
)


# Entrée du décodeur :
# un caractère précédent pour chaque phrase du batch

input_token = torch.tensor(
    [1, 1, 1, 1]
)


# Etat caché venant de l'encodeur

hidden = torch.randn(
    1,
    BATCH_SIZE,
    HIDDEN_SIZE
)


# Vecteur contexte venant de l'attention

context = torch.randn(
    BATCH_SIZE,
    1,
    HIDDEN_SIZE
)


# Passage dans le décodeur

prediction, hidden = decoder(
    input_token,
    hidden,
    context
)


print("Sortie prédiction :")
print(prediction.shape)


print("\nNouvel état caché :")
print(hidden.shape)