from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import TFIDF_CONFIG


class TFIDFVectorizer:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(

            **TFIDF_CONFIG

        )

    def fit(self, texts):

        self.vectorizer.fit(texts)

        return self

    def transform(self, texts):

        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):

        return self.vectorizer.fit_transform(texts)

    def get_feature_names(self):

        return self.vectorizer.get_feature_names_out()

    def get_model(self):

        return self.vectorizer