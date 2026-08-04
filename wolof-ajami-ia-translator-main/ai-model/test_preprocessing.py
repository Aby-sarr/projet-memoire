from config.config import CORPUS_FILE
from utils.preprocessing import CorpusPreprocessor
from pathlib import Path

preprocessor = CorpusPreprocessor(CORPUS_FILE)
df = preprocessor.load_corpus()

preprocessor.check_columns(df)
columns = preprocessor.check_columns(df)

df = preprocessor.clean_corpus(
    df,
    columns[0],
    columns[1]
)
output = Path("data/corpus_cleaned.csv")

preprocessor.save_cleaned_corpus(
    df,
    output
)