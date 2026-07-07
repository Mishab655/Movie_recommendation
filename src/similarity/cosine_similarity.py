from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarityEngine:

    def __init__(self):
        self.similarity_matrix = None

    def fit(self, vectors):
        """
        Compute pairwise cosine similarity.
        """
        self.similarity_matrix = cosine_similarity(vectors)
        return self

    def get_similarity_matrix(self):
        return self.similarity_matrix

    def get_similar_movies(self, movie_index, top_k=10):

        scores = list(enumerate(self.similarity_matrix[movie_index]))

        scores = sorted(
            scores,
            key=lambda x: x[1],
            reverse=True
        )

        return scores[1:top_k+1]