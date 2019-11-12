import requests
import pandas as pd
import numpy as np
import json
from bs4 import BeautifulSoup
import config
import logging
import importlib
import time
import random
import validators
import pickle
import time
import random
from calendar import monthrange
import re
import spacy
from spacy.tokens import Doc, Span
import nltk
from nltk.corpus import stopwords
from nltk.corpus import treebank
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import wordnet

from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

import string
import unicodedata

import matplotlib.pyplot as plt  

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as datasets
import pandas as pd
import numpy as np
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import mean_squared_error
from pandas.plotting import scatter_matrix
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn import tree 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC  
from time import time
import warnings
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer


import gensim
from gensim import corpora
from gensim import models
from gensim.sklearn_api import lsimodel, ldamodel
from gensim.matutils import sparse2full
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.coherencemodel import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim


class TextNormalizer(BaseEstimator, TransformerMixin):
    """
    TextNormalizer(BaseEstimator, TransformerMixin):
    This class packages up key NLP processing steps
    for a news classification project and is meant to be able
    to be used in an SKLeanr pipeline.  However it also functions
    as a standalone class and I have used it in this way.
    Key steps are:
    - remove stop words, including "custom" stop words (i.e., words
        in news texts that I deemed of low value);
    - tokenization
    """
    def __init__(self, language='english', source=None):
        self.source = source
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.more_stops = {'mr', 'ms', 'mr.', 'ms.', 'mrs', 'mrs.', 'say',
                            'said', 'saying', 'also', 'yeh', 'hom', 'even',
                            'like', 'k', 'n', 'u', 'would', 'could', '$',
                            'ft', 'per','cent'}
        self.stopwords.update(self.more_stops)
        self.lemmatizer = WordNetLemmatizer()

    def is_punct(self, token):
        return all(unicodedata.category(char).startswith('P') for char in token)

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def get_stopwords(self):
        return self.stopwords

    def get_wordnet_pos(selfm, word):
        """
        Get a POS tag for the given word, then convert
        to the appropriate tag for NLTK's lemmatize function
        """
        # Map POS tag to first character lemmatize()
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    def normalize(self, document):
        """
        Main work to normalize text
        Params:
            document:  a single instane of text - i.e., single news article
        Returns:
            norm_toks - document converted to normalized tokens, in list form
        """
        NYT_PAT = '^[A-Z]* — '
        NYT_COMP_PAT = re.compile(NYT_PAT)

        norm_toks = []

        if self.source and self.source == 'NYT':
            document = NYT_COMP_PAT.sub('', document)
# I found nltk's sentenizer to perform poorly
#        for sent in sent_tokenize(document):
        for sent in document.split('.'):
            for tok in nltk.word_tokenize(sent):
                if self.is_punct(tok) or tok.isdigit():
                    continue
                lem = self.lemmatize(tok, self.get_wordnet_pos(tok)).lower()
                if not self.is_stopword(lem):
                    norm_toks.append(lem)

        return norm_toks

    def lemmatize(self, token, pos):
        """
        Invoke the WordNet lemmatizer (from NLTK)
        Params:
            token:  individual word token
            pos:    wordnet POS tag
        Returns:
            lemmatized word (token)
        """
        return self.lemmatizer.lemmatize(token, pos)

    def fit(self, X, y=None):
        """
        Override SKLearn method
        """
        return self

    def transform(self, documents):
        """
        Override SKLearn method.
        Do the main work of normalizing the text
        Params:
            documents:  full list of documents - corpus.
        Returns:
            normalized list of tokens
       """
        for doc in documents:
            yield self.normalize(doc)


class BiGramTransformer:
    """
    Custom bigram transformer.  Based on domain of world news,
    look for specific instances of names (and places) and create
    custom bigrams - that is, word tokens comprising two words
    """
    
    def __init__(self):
        self.bigram_list = ['donald trump', 'angela merkel', 'boris johnson', 'vladimir putin', 'benjamin netanyahu',
                           'hong kong', 'north korea', 'south korea', 'united kingdom', 'united states', 'south africa',
                            'xi jinping', 'carrie lam', 'oleksiy honcharuk', 'volodymyr zelensky', 'emmanuel macron',
                           'viktor orbán', 'justin trudeau', 'north america', 'south america', 'sinn fein', 'jeremy corbyn', 'narendra modi',
                            'mohamed morsi', 'shuping wang', 'hassan rouhani', 'rudy giuliani', 'joe biden', 'new zealand', 'european union', 'pope francis']

        self.first_grams = [word.split()[0] for word in self.bigram_list]
        self.second_grams = [word.split()[1] for word in self.bigram_list]

    def get_custom_bigrams(self, tokens):
        """
        Do the work to find instamces of our custom bigrams
        in a list of tokens and re-format as appropriate
        Params:
            tokens:  list of word tokens
        Returns
            new list of tokens, including any custom bigrams
        """
        new_toks = []
        length = len(tokens)

        for ix, tok in enumerate(tokens):
            try:
                if tok in self.first_grams:
                    if ix + 1 <= length and tokens[ix + 1] in self.second_grams:
                        new_toks.append(' '.join([tok, tokens[ix+1]]))
                        continue
                if tok in self.second_grams:
                    # we assume if we see the 2nd token in a bigram that
                    # we've already prcessed the first and second together
                    # so skip
                    continue
                new_toks.append(tok)
            except IndexError:
                continue

        return new_toks


def get_kanji_text_rows(df):
    """
    Free-standing helper function:  detect kanji text rows in a dataframe
    Params:
        df:  entire dataframe of news articles
    Returns:
        Specific rows of DF that contain kanji - these rows will be used
        by the drop() function.
    """

    kanji_pat = re.compile(u'[\u4e00-\u9fff]')
    kanji_rows = []
    for i in range(0, len(df)):
        text = df.iloc[i].text
        if type(text) is not str:
            continue
        if kanji_pat.search(df.iloc[i].text):
            kanji_rows.append(i)

    return kanji_rows
