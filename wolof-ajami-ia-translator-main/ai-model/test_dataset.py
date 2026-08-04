from utils.dataset import WolofAjamiDataset
from utils.vocabulary import Vocabulary


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


print("Nombre de données :", len(dataset))


source, target = dataset[0]


print("\nSource :")
print(source)

print("\nCible :")
print(target)