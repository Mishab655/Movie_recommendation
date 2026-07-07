from pathlib import Path

# -------------------------------
# Project Paths
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "final_movies.csv"

MODEL_DIR = BASE_DIR / "models"

TFIDF_MODEL_PATH = MODEL_DIR / "tfidf" / "tfidf.pkl"
TFIDF_VECTOR_PATH = MODEL_DIR / "tfidf" / "vectors.npz"

COUNT_MODEL_PATH = MODEL_DIR / "count" / "count.pkl"
COUNT_VECTOR_PATH = MODEL_DIR / "count" / "vectors.npz"

HASHING_MODEL_PATH = MODEL_DIR / "hashing" / "hashing.pkl"
HASHING_VECTOR_PATH = MODEL_DIR / "hashing" / "vectors.npz"

WORD2VEC_MODEL_PATH = MODEL_DIR / "word2vec" / "word2vec.model"
WORD2VEC_VECTOR_PATH = MODEL_DIR / "word2vec" / "embeddings.npy"

TFIDF_SIMILARITY_PATH = (
    MODEL_DIR / "tfidf" / "similarity.npy"
)

COUNT_SIMILARITY_PATH = (
    MODEL_DIR / "count" / "similarity.npy"
)

HASHING_SIMILARITY_PATH = (
    MODEL_DIR / "hashing" / "similarity.npy"
)

WORD2VEC_SIMILARITY_PATH = (
    MODEL_DIR / "word2vec" / "similarity.npy"
)

# -------------------------------
# Selected Vectorizer
# -------------------------------

# VECTORIZER = "tfidf"
# VECTORIZER = "count"
# VECTORIZER = "hashing"
VECTORIZER = "word2vec"

# ===============================
# Similarity Engine
# ===============================

SIMILARITY_ENGINE = "cosine"

"""
Options

cosine
dot
faiss
"""
# -------------------------------
# TF-IDF Parameters
# -------------------------------

TFIDF_CONFIG = {

    "stop_words": "english",

    "ngram_range": (1,2),

    "min_df": 3,

    "max_features": 10000,

    "strip_accents": "unicode",

    "sublinear_tf": True,

    "smooth_idf": True,

    "norm": "l2"

}

# ===============================
# CountVectorizer Configuration
# ===============================

COUNT_CONFIG = {

    "stop_words": "english",

    "ngram_range": (1, 2),

    "min_df": 3,

    "max_features": 10000

}

# ===============================
# HashingVectorizer
# ===============================

HASH_CONFIG = {

    "n_features": 2**18,

    "alternate_sign": False,

    "norm": "l2",

    "stop_words": "english",

    "ngram_range": (1,2)

}

# ===============================
# Word2Vec
# ===============================

WORD2VEC_CONFIG = {

    "vector_size": 300,

    "window": 5,

    "min_count": 2,

    "workers": 4,

    "epochs": 20,

    "sg": 1

}

