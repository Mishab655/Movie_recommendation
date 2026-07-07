import pandas as pd

from src.config import *
from src.model_saver import ModelSaver
from src.recommender import RecommendationEngine


df = pd.read_csv(DATA_PATH)

similarity_matrix = ModelSaver.load_similarity(
    MODEL_DIR / VECTORIZER / "similarity.npy"
)

engine = RecommendationEngine(
    movies_df=df,
    similarity_matrix=similarity_matrix
)