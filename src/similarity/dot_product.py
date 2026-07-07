import numpy as np


class DotProductSimilarityEngine:

    def __init__(self):
        self.similarity_matrix = None

    def fit(self, vectors):

        self.similarity_matrix = np.dot(
            vectors,
            vectors.T
        )

        return self

    def get_similarity_matrix(self):

        return self.similarity_matrix

    def get_similar_movies(self, movie_index, top_k=10):

        scores = list(
            enumerate(
                self.similarity_matrix[movie_index]
            )
        )

        scores = sorted(
            scores,
            key=lambda x: x[1],
            reverse=True
        )

        return scores[1:top_k+1]