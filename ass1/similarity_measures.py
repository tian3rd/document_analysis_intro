from abc import abstractmethod
from collections import defaultdict
from math import log, sqrt
import math
import re
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
        '''
        document_term_freq: number of occurence of term t / (token) in document d / (doc)
        '''
        for token, query_term_frequency in query.items():
            # print("token: {0}, query_term_frequency: {1}".format(token, query_term_frequency))
            for doc, document_term_frequency in self.postings.token_to_doc_counts[token].items():
                # print("----------------------------------------------------------------")
                # print("token: {0}, query_term_frequency: {1}".format(token, query_term_frequency))
                # print("doc: {0}, document_term_frequency: {1}".format(doc, document_term_frequency))
                doc_to_score[doc] += query_term_frequency * document_term_frequency / self.doc_to_norm[doc]

class TFIDF_Similarity(CosineSimilarity):
    # TODO implement the set_document_norms and get_scores methods.
    # Get rid of the NotImplementedErrors when you are done.
    # def __init__(self, postings):
    #     # could try using **kargs and setattr to pass in the modes...
    #     super().__init__(postings)
    #     self.TF_mode = None
    #     self.DF_mode = None
    #     self.Norm_mode = None
    #     # total number of docs
    #     self.N = len(self.postings.doc_to_token_counts)

    # set norm by mode
    def set_document_norms(self):
        def set_modes(tf_mode, df_mode, norm_mode):
            '''
            TF_MODES = ["n", "l", "a", "b", "L"]
            DF_MODES = ["n", "t", "p"]
            NORM_MODES = ["n", "c", "u", "b"]
            '''
            self.TF_mode = tf_mode
            self.DF_mode = df_mode
            self.Norm_mode = norm_mode
            # total number of docs
            self.N = len(self.postings.doc_to_token_counts)
        
        set_modes("l", "n", "c")

        doc_lengths = dict()
        # c (cosine) norm
        if self.Norm_mode == "c":
            for doc, token_counts in self.postings.doc_to_token_counts.items():
                self.doc_to_norm[doc] = sqrt(sum([tf ** 2 for tf in token_counts.values()])) 
                doc_lengths[doc] = sum([tf for tf in token_counts.values()])
        # n (none) norm
        if self.Norm_mode == "n":
            for doc, token_counts in self.postings.doc_to_token_counts.items():
                self.doc_to_norm[doc] = 1
        # b (byte size)
        if self.Norm_mode == "b":
            alpha = .5
            for doc, token_counts in self.postings.doc_to_token_counts.items():
                self.doc_to_norm[doc] = sum([(len(term)*tf) for term, tf in token_counts.items()])**alpha
        # u (pivoted unique)

        # find out longest doc
        print("Longest doc lenght: {0}".format(max(doc_lengths.values())))

    def get_scores(self, doc_to_score, query):
        for token, query_term_frequency in query.items():
            # all term frequencies for token/term in all documents -> a list of frequencies
            tf_all = list(self.postings.token_to_doc_counts[token].values())
            # print("tf_all 1st values: {0}".format(tf_all[0]))
            for doc, document_term_frequency in self.postings.token_to_doc_counts[token].items():
                # original term frequency
                tf = document_term_frequency
                # print("tf: {0}, tf mode: {1}".format(tf, self.TF_mode))
                # document frequency of term t / token
                df = len(self.postings.token_to_doc_counts[token])
                # print("df length: {0}".format(df))
                # sef tf by TF_mode
                tf = self.set_tf(tf, tf_all, self.TF_mode)
                # print("tf after: {0}".format(tf))
                # set df by DF_mode
                df = self.set_df(df, self.DF_mode)
                # print("df length after: {0}".format(df))
                doc_to_score[doc] += query_term_frequency * tf * df / self.doc_to_norm[doc]

    def set_tf(self, tf, tf_all, mode):
        if mode == "n":
            # print("tf mode: {0}".format(mode))
            return tf
        if mode == "l":
            return 1 + math.log(tf)
        if mode == "a":
            tf_max = max(tf_all)
            return .5 + .5 * tf / tf_max
        if mode == "L":
            tf_ave = sum(tf_all) / len(tf_all)
            return (1+math.log(tf)) / (1+math.log(tf_ave))
    
    def set_df(self, df, mode):
        if mode == "n":
            return 1
        if mode == "t":
            return math.log(self.N / df)
        if mode == "p":
            return max(0, math.log((self.N-df)/df))
