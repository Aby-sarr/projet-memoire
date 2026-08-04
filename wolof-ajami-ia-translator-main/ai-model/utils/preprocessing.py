import pandas as pd
from pathlib import Path


class CorpusPreprocessor:
    """
    Classe responsable du chargement et du prétraitement
    du corpus Wolof ↔ Ajami.
    """

    def __init__(self, corpus_path):
        self.corpus_path = Path(corpus_path)


    def load_corpus(self):
        """
        Charge le corpus CSV.
        """

        if not self.corpus_path.exists():
            raise FileNotFoundError(
                f"Le fichier {self.corpus_path} est introuvable."
            )

        df = pd.read_csv(self.corpus_path)

        if df.empty:
            raise ValueError("Le corpus est vide.")

        print("=" * 50)
        print("Corpus chargé avec succès")
        print(f"Nombre de paires : {len(df)}")
        print("=" * 50)

        return df


    def check_columns(self, df):
        """
        Vérifie les colonnes disponibles dans le corpus.
        """

        print("\nColonnes du corpus :")
        print(list(df.columns))

        return list(df.columns)
    
      # Nettoyage du corpus
    def clean_corpus(self, df, wolof_col, ajami_col):
        """
        Nettoyage du corpus.
        """

        print("\n==============================")
        print("Nettoyage du corpus")
        print("==============================")

        initial_size = len(df)

        df = df.dropna(subset=[wolof_col, ajami_col])

        empty_removed = initial_size - len(df)

        df[wolof_col] = df[wolof_col].astype(str).str.strip()
        df[ajami_col] = df[ajami_col].astype(str).str.strip()

        before_duplicates = len(df)

        df = df.drop_duplicates(
            subset=[wolof_col, ajami_col]
        )

        duplicates_removed = before_duplicates - len(df)

        print(f"Avant nettoyage : {initial_size}")
        print(f"Lignes vides supprimées : {empty_removed}")
        print(f"Doublons supprimés : {duplicates_removed}")
        print(f"Après nettoyage : {len(df)}")

        print("==============================")

        return df

    # Sauvegarde du corpus nettoyé
    def save_cleaned_corpus(self, df, output_path):
        """
        Sauvegarde le corpus nettoyé.
        """

        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8"
        )

        print(f"\nCorpus nettoyé sauvegardé : {output_path}")