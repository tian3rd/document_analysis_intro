from abc import abstractmethod
from collections import defaultdict
from math import log, sqrt
import numpy as np


class CosineSimilarity:
    """
    This class calculates a similarity score between a given query and all documents in an inverted index.
    """
    def __init__(self, postings):
        self.postings = postings
        self.doc_to_norm = dict()
        self.set_document_norms()

    def __call__(self, query):
        doc_to_score = defaultdict(lambda: 0)
        self.get_scores(doc_to_score, query)
        return doc_to_score

    @abstractmethod
    def set_document_norms(self):
        """
        Set self.doc_to_norm to contain the norms of every document.
        """
        pass

    @abstractmethod
    def get_scores(self, doc_to_score, query):
        """
        For each document add an entry to doc_to_score with this document's similarity to query.
        """
        pass


class TF_Similarity(CosineSimilarity):
    def set_document_norms(self):
        for doc, token_counts in self.postings.doc_to_token_counts.items():
            self.doc_to_norm[doc] = sqrt(sum([tf ** 2 for tf in token_counts.values()]))

    def get_scores(self, doc_to_score, query):
        for token, query_term_frequency in query.items():
            for doc, document_term_frequency in self.postings.token_to_doc_counts[token].items():
                doc_to_score[doc] += query_term_frequency * document_term_frequency / self.doc_to_norm[doc]

class TFIDF_Similarity(CosineSimilarity):
    # TODO implement the set_document_norms and get_scores methods.
    # Get rid of the NotImplementedErrors when you are done.
    def set_document_norms(self):
        raise NotImplementedError

    def get_scores(self, doc_to_score, query):
        raise NotImplementedError


