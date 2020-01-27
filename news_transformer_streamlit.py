#!/usr/bin/env python
# coding: utf-8

# In[25]:


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
from selenium import webdriver
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from selenium.common.exceptions import TimeoutException
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

pd.options.display.max_columns = 300
import string
import unicodedata

pd.options.display.max_columns = 200
pd.options.display.max_rows = 999
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
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.svm import SVC  
from time import time
np.random.seed(0)
import warnings
warnings.filterwarnings('ignore')

from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


import gensim
from gensim import corpora
from gensim import models
from gensim.sklearn_api import lsimodel, ldamodel
from gensim.matutils import sparse2full
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel

import pyLDAvis
import pyLDAvis.gensim

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from functools import partial
from operator import itemgetter

from joblib import dump, load
import streamlit as st
from PIL import Image


# In[7]:


class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, extra_stop_words, bigram_list, language='english', source=None):
        self.source = source
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.more_stops = extra_stop_words
        self.stopwords.update(self.more_stops) 
        self.lemmatizer = WordNetLemmatizer()
        
        self.bigram_list = bigram_list        
        self.first_grams = [word.split()[0] for word in bigram_list]
        self.second_grams = [word.split()[1] for word in bigram_list]

    def get_custom_bigrams(self, tokens):
        new_toks = []
        length = len(tokens)

        for ix, tok in enumerate(tokens):
            try:
                if tok in self.first_grams:
                    if ix+1 <= length and tokens[ix+1] in self.second_grams:
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

    def is_punct(self, token):
        return all(unicodedata.category(char).startswith('P') for char in token)

    def is_stopword(self, token):
        return token.lower() in self.stopwords
    
    def get_stopwords(self):
        return self.stopwords

    def get_wordnet_pos(selfm, word):
        #Map POS tag to first character lemmatize() 
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    def normalize(self, document):
        NYT_PAT = '^[A-Z]* — '
        NYT_COMP_PAT = re.compile(NYT_PAT)        
        
        norm_toks = []
    
        if self.source and self.source == 'NYT':
            document = NYT_COMP_PAT.sub('', document)
        
        if self.source and self.source == 'RT':
            document = self.rt_clean(document)
            
        # NOTE:  I found nltk's sentenizer to perform poorly
            #        for sent in sent_tokenize(document):
        for sent in document.split('.'):            
            for tok in nltk.word_tokenize(sent):
                if self.is_punct(tok) or tok.isdigit():
                    continue
                lem = self.lemmatize(tok, self.get_wordnet_pos(tok)).lower()
                if not self.is_stopword(lem):
                    norm_toks.append(lem) 
#         bigram_toks = self.get_custom_bigrams(norm_toks)
#         return ' '.join(bigram_toks)
#         return ' '.join(norm_toks)       
        return ' '.join(self.get_custom_bigrams(norm_toks))
    
    def rt_clean(self, text):
        #
        # first match this simple, typical Reuters pattern:
        # "(Reuters) - British Prime Minister Boris Johnson..."
        rt_pat = r'^\(Reuters\) - '
        RT_PAT = re.compile(rt_pat)
        text = RT_PAT.sub('', text)

        rt_pat2 = r'^\(This.*\)'

        RT_PAT2 = re.compile(rt_pat2)
        text = RT_PAT2.sub('', text)

        # match this:  "ANKARA/Turkish-backed Syrian...""
        # or this:  "MAMALLAPURAM, India, Chinese President...""
        # or this:  "ADDIS For Samson Berhane, the news that...""
        caps_loc_pat = r'[A-Z]{2,20}[ \/,]{1}'
        CAPS_PAT = re.compile(caps_loc_pat)

        # After removing first pattern, 
        # match this:  "IndiaAt least five people were injured in a grenade attack"
        # but save "At least..."
        name_against_text_pat = r'^ *[A-Z][a-z]{2,20}([A-Z]{1})'
        NM_TEXT_PAT = re.compile(name_against_text_pat)

        first_sub = CAPS_PAT.sub('', text)

        match = NM_TEXT_PAT.match(first_sub)
        rep_char = ''
        if match:
            rep_char = match[1]

        t = NM_TEXT_PAT.sub('', first_sub)
        # now add back that lone capital letter
        text = rep_char + t

        return text    


    def lemmatize(self, token, pos):
        return self.lemmatizer.lemmatize(token, pos)

    def fit(self, X, y=None):
        self.source = y
        return self

    def transform(self, documents):
        for doc in documents:
            yield self.normalize(doc)


# In[15]:


MODEL_PATH ='/Users/markbrennan/nlp_news_classifier/objects/'
MODEL_NAME = 'dt-model-new-bi.joblib'
TFIDF_VEC_NAME = 'tfidf-new-bi.joblib'


# In[8]:


# code to pull out our custom bi-grams
bigrams = ['donald trump', 'angela merkel', 'boris johnson', 'vladimir putin', 'benjamin netanyahu', 
           'hong kong', 'north korea', 'south korea', 'united kingdom', 'united states', 'south africa', 
           'xi jinping', 'carrie lam', 'oleksiy honcharuk', 'volodymyr zelensky', 'emmanuel macron',
           'viktor orbán', 'justin trudeau', 'north america', 'south america', 'sinn fein', 'jeremy corbyn', 'narendra modi',
            'mohamed morsi', 'shuping wang', 'hassan rouhani', 'rudy giuliani', 'joe biden', 'new zealand', 'european union', 'pope francis']


# In[9]:


more_stops = {'mr', 'ms', 'mr.', 'ms.', 'mrs', 'mrs.','say', 'said', 'saying', 
                   'also', 'yeh', 'hom', 'even', 'like', 'k', 'n', 'u', 'would', 'could', '$', 'ft', 'per','cent'}
        

# In[21]:


def create_and_run_pipeline(doc, model, vector, extra_stops=more_stops, bigrams=bigrams, source=None):
    textNormer = TextNormalizer(extra_stop_words=more_stops, bigram_list=bigrams, source=source)
    text = textNormer.normalize(doc)
    text_series = pd.Series(text)
    text_vec = vector.transform(text_series)
    train_pred = model.predict(text_vec.toarray())
    return train_pred[0]


# In[11]:


def classify_text(text):
    # load the model
    full_model_name = MODEL_PATH + MODEL_NAME
    model = joblib.load(full_model_name)    

    # load the TF-IDF Vector
    full_vector_name = MODEL_PATH + TFIDF_VEC_NAME
    vector = joblib.load(full_vector_name)

    # run the full pipeline on the input text with the vector and model
    result = create_and_run_pipeline(text, model, vector, extra_stops=more_stops, bigrams=bigrams)

    return result



rt_image = Image.open('reuters.png')
ft_image = Image.open('ft.png')
nyt_image = Image.open('nyt.png')

image_dict = {'NYT': nyt_image, 'FT': ft_image, 'RT': rt_image}

text = None

st.title('NLP News Classifier')

st.text('\n')
st.text('\n')
st.text('\n')

st.markdown('**Enter text of article to classify**')
text = st.text_area('')

if text:
    pred_result = classify_text(text)
    st.text('\n')
    st.write('Prediction:')
    st.image(image_dict[pred_result],  use_column_width=False, format='PNG')
#st.write(f'Article is classified as {pred_result[0]}')





from nyt_scraping_script import get_nyt_article
def fetch_nyt_article(url):
    try:
        if validators.url(url) == True:
            response = requests.get(url)
            # If the response was successful, no Exception will be raised
            response.raise_for_status()
        else:
            logging.exception(f'url is invalid!  url:  {url}')
            return None
    except requests.exceptions.HTTPError as http_err:
        logging.exception(f'HTTP error occurred: {http_err} - url: {url}')  # Python 3.6
        return None
    except Exception as err:
        logging.exception(f'Other error occurred: {err} - url: {url}')  # Python 3.6
        return None
                                    
    # success - do the text extraction!
    soup = BeautifulSoup(response.content, 'html.parser')
    return get_nyt_article(soup)

