from gensim.models import Word2Vec

import numpy as np

from src.config import WORD2VEC_CONFIG


class Word2VecVectorizer:

    def __init__(self):

        self.model = None
        
    def fit(self, texts):

        tokenized = [
            text.split()
            for text in texts
        ]

        self.model = Word2Vec(
            sentences=tokenized,
            **WORD2VEC_CONFIG
        )

        return self
    
    def transform(self, texts):

        tokenized = [

            text.split()

            for text in texts

        ]

        movie_vectors = []

        for sentence in tokenized:

            vectors = [

                self.model.wv[word]

                for word in sentence

                if word in self.model.wv

            ]

            if len(vectors):

                movie_vectors.append(

                    np.mean(

                        vectors,

                        axis=0

                    )

                )

            else:

                movie_vectors.append(

                    np.zeros(

                        self.model.vector_size

                    )

                )

        return np.array(movie_vectors)
    
    def fit_transform(self, texts):

        self.fit(texts)

        return self.transform(texts)
    
    def get_model(self):

        return self.model