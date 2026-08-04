from pathlib import Path
import torch
# la racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# Dossiers
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"

# fichiers
CORPUS_FILE = DATA_DIR / "corpus_cleaned.csv"
WOLOF_VOCAB_FILE = DATA_DIR / "vocab_wolof.json"
AJAMI_VOCAB_FILE = DATA_DIR / "vocab_ajami.json"

# Hyperparamètres
BATCH_SIZE = 32
EMBEDDING = 256
HIDDEN_SIZE = 512
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
TEACHER_FORCING_RATIO = 0.5
EARLY_STOPPING_PATIENCE = 3

# utiliser le GPU si disponible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")