from sklearn.feature_extraction.text import HashingVectorizer

from src.config import HASH_CONFIG


class HashingVectorizerModel:

    def __init__(self):

        self.vectorizer = HashingVectorizer(

            **HASH_CONFIG

        )

    def fit(self, texts):

        """
        HashingVectorizer doesn't learn
        a vocabulary.

        So fit() does nothing.
        """

        return self

    def transform(self, texts):

        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):

        return self.transform(texts)

    def get_model(self):

        return self.vectorizer