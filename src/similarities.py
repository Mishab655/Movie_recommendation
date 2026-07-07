from src.similarity.cosine_similarity import (
    CosineSimilarityEngine
)

from src.similarity.dot_product import (
    DotProductSimilarityEngine
)


def get_similarity_engine(name):

    name = name.lower()

    if name == "cosine":
        return CosineSimilarityEngine()

    elif name == "dot":
        return DotProductSimilarityEngine()

    else:
        raise ValueError(
            f"Unknown similarity engine: {name}"
        )