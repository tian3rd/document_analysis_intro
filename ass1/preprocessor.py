import nltk
from functools import lru_cache


class Preprocessor:
    def __init__(self):
        # Stemming is the most time-consuming part of the indexing process, we attach a lru_cache to the stermmer
        # which will store upto 100000 stemmed forms and reuse them when possible instead of applying the
        # stemming algorithm.
        self.stem = lru_cache(maxsize=100000)(nltk.PorterStemmer().stem)
        self.tokenize = nltk.tokenize.WhitespaceTokenizer().tokenize

    def __call__(self, text):
        text = text.lower()
        tokens = nltk.WhitespaceTokenizer().tokenize(text)
        tokens.extend([self.stem(token) for token in tokens])
        # TODO by me: may add other funcs like usa == u.s.a == united states

        return tokens
