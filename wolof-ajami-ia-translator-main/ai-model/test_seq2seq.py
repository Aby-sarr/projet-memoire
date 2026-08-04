import torch

from models.encoder import EncoderGRU
from models.decoder import DecoderGRU
from models.attention import BahdanauAttention
from models.seq2seq import Seq2Seq


INPUT_DIM = 80
OUTPUT_DIM = 51

EMBEDDING_DIM = 256
HIDDEN_SIZE = 512


device = torch.device("cpu")


encoder = EncoderGRU(
    INPUT_DIM,
    EMBEDDING_DIM,
    HIDDEN_SIZE
)


attention = BahdanauAttention(
    HIDDEN_SIZE
)


decoder = DecoderGRU(
    OUTPUT_DIM,
    EMBEDDING_DIM,
    HIDDEN_SIZE
)


model = Seq2Seq(
    encoder,
    decoder,
    attention,
    device
)


src = torch.randint(
    0,
    INPUT_DIM,
    (4,10)
)


trg = torch.randint(
    0,
    OUTPUT_DIM,
    (4,12)
)


output = model(
    src,
    trg
)


print("Sortie modèle :")
print(output.shape)