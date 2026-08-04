from utils.dataset import WolofAjamiDataset
from utils.vocabulary import Vocabulary
from utils.dataloader import create_dataloader


wolof_vocab = Vocabulary(
    "data/vocab_wolof.json"
)

ajami_vocab = Vocabulary(
    "data/vocab_ajami.json"
)


dataset = WolofAjamiDataset(
    "data/corpus_cleaned.csv",
    wolof_vocab,
    ajami_vocab
)


loader = create_dataloader(
    dataset,
    batch_size=4,
    pad_idx=0
)


source, target = next(iter(loader))


print("Taille source :", source.shape)
print("Taille cible :", target.shape)


print("\nSource batch :")
print(source)

print("\nTarget batch :")
print(target)