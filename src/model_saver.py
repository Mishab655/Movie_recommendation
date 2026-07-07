import joblib
import numpy as np
import scipy.sparse as sp

from pathlib import Path

from gensim.models import Word2Vec

class ModelSaver:

    # ============================
    # Generic Models (scikit-learn)
    # ============================

    @staticmethod
    def save_model(model, filepath):

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(model, filepath)

    @staticmethod
    def load_model(filepath):

        return joblib.load(filepath)

    # ============================
    # Sparse Matrices
    # ============================

    @staticmethod
    def save_sparse_matrix(matrix, filepath):

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        sp.save_npz(filepath, matrix)

    @staticmethod
    def load_sparse_matrix(filepath):

        return sp.load_npz(filepath)

    # ============================
    # NumPy Arrays
    # ============================

    @staticmethod
    def save_numpy(array, filepath):

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(filepath, array)

    @staticmethod
    def load_numpy(filepath):

        return np.load(filepath)

    # ============================
    # Word2Vec
    # ============================

    @staticmethod
    def save_word2vec(model, filepath):

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        model.save(str(filepath))

    @staticmethod
    def load_word2vec(filepath):

        return Word2Vec.load(str(filepath))
    
    
    @staticmethod
    def save_similarity(matrix, filepath):

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(filepath, matrix)


    @staticmethod
    def load_similarity(filepath):

        return np.load(filepath)