from src.vectorizer.tfidf_vectorizer import TFIDFVectorizer
from src.vectorizer.count_vectorizer import CountVectorizerModel
from src.vectorizer.hashing_vectorizer import HashingVectorizerModel
from src.vectorizer.word2vec_vectorizer import Word2VecVectorizer

def get_vectorizer(name):

    name = name.lower()

    if name == "tfidf":

        return TFIDFVectorizer()
    
    elif name == "count":

        return CountVectorizerModel()
    
    elif name == "hashing":

        return HashingVectorizerModel()
    
    elif name == "word2vec":

        return Word2VecVectorizer()
    
    raise ValueError(

        f"{name} not supported."

    )