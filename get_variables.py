__author__ = 'vinay_vijayan'

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.externals import joblib

from sklearn import preprocessing
import pickle

from scipy import sparse
from os.path import isfile

import time

# Get an instance of a logger
import logging
logger = logging.getLogger(__name__)

class VariablesXandY(object):
    def __init__(self, input_filename=None):
        if isinstance(input_filename, pd.DataFrame) or not input_filename:
            self.df_whole_data = input_filename
        elif isfile(input_filename):
            self.df_whole_data = pd.read_csv(input_filename, sep=',', quotechar='"', encoding='utf-8')
        else:
            raise ValueError("Input must be filepath or dataframe!")

    def get_x_matrix(self, ngram_range, x_pickle_filename=None):
        x_train = self.df_whole_data['text']

        if x_pickle_filename == None:
            x_train_tfidf = VariablesXandY.get_x(x_train, ngram_range)
        else:
            if isfile(x_pickle_filename):
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ' x_train.pkl found ')
                x_train_tfidf = pickle.load(open(x_pickle_filename, 'r'))
            else:
                x_train_tfidf = VariablesXandY.get_x(x_train, ngram_range)
                pickle.dump(x_train_tfidf, open(x_pickle_filename, 'w'))

        return x_train_tfidf


    def get_y_matrix(self, labels_pickle_filename=None, y_pickle_filename=None):
        def __get_y():
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ' Starting y_train ')
            codes_train = self.df_whole_data.ix[:, 'act_code1':'act_code33']

            y_train = []
            for index, row in codes_train.iterrows():
                row = row[np.logical_not(np.isnan(row))].astype(str)
                row = tuple(row.tolist())
                y_train.append(row)

            tuple_unique_labels = tuple(set(x for l in y_train for x in l))
            label_binarizer_object = preprocessing.MultiLabelBinarizer(classes=tuple_unique_labels)
            y_train_binarized = sparse.csr_matrix(label_binarizer_object.fit_transform(y_train))

            self.fitted_labels = label_binarizer_object.fit(y_train)

            if labels_pickle_filename:
                joblib.dump(self.fitted_labels, labels_pickle_filename)

            return y_train_binarized

        if y_pickle_filename == None:
            y_train_binarized = __get_y()
        else:
            if isfile(y_pickle_filename):
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ' y_train.pkl found ')
                y_train_binarized = pickle.load(open(y_pickle_filename, 'r'))
            else:
                y_train_binarized = __get_y()
                pickle.dump(y_train_binarized, open(y_pickle_filename, 'w'))
        return y_train_binarized

    @staticmethod
    def get_x(text,ngram_range):

        hash_vect_object = HashingVectorizer(ngram_range=ngram_range,
                                             stop_words="english",
                                             strip_accents="unicode",
                                             token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z_]+\b") # tokens are character strings of 2 or more characters
        tfidf_transformer_object = TfidfTransformer(use_idf=True)

        x_train_counts = hash_vect_object.fit_transform(text)
        x_train_tfidf = tfidf_transformer_object.fit_transform(x_train_counts)

        return x_train_tfidf
