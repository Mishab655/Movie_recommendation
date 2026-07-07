import pandas as pd

from src.config import *
from src.vectorizers import get_vectorizer
from src.similarities import get_similarity_engine
from src.model_saver import ModelSaver

# ====================================
# Load Dataset
# ====================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

texts = df["combined_features"].fillna("").astype(str)

# ====================================
# Train Vectorizer
# ====================================

print(f"Training {VECTORIZER}...")

vectorizer = get_vectorizer(VECTORIZER)

vectors = vectorizer.fit_transform(texts)

# ====================================
# Save Vectorizer
# ====================================

print("Saving vectorizer...")

if VECTORIZER in ["tfidf", "count", "hashing"]:

    ModelSaver.save_model(
        vectorizer.get_model(),
        MODEL_DIR / VECTORIZER / f"{VECTORIZER}.pkl"
    )

    ModelSaver.save_sparse_matrix(
        vectors,
        MODEL_DIR / VECTORIZER / "vectors.npz"
    )

elif VECTORIZER == "word2vec":

    ModelSaver.save_word2vec(
        vectorizer.get_model(),
        WORD2VEC_MODEL_PATH
    )

    ModelSaver.save_numpy(
        vectors,
        WORD2VEC_VECTOR_PATH
    )

else:
    raise ValueError(f"Unsupported vectorizer: {VECTORIZER}")

# ====================================
# Build Similarity Matrix
# ====================================

print(f"Building {SIMILARITY_ENGINE} similarity...")

similarity_engine = get_similarity_engine(
    SIMILARITY_ENGINE
)

similarity_engine.fit(vectors)

similarity_matrix = similarity_engine.get_similarity_matrix()

# ====================================
# Save Similarity Matrix
# ====================================

print("Saving similarity matrix...")

similarity_path = MODEL_DIR / VECTORIZER / "similarity.npy"

ModelSaver.save_similarity(
    similarity_matrix,
    similarity_path
)

print("Training completed successfully.")