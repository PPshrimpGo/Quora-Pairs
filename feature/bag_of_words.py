#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import math
import nltk
import datetime
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.optimize import minimize
stops = set(stopwords.words("english"))
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import multiprocessing
import difflib
import re
from string import punctuation
import sys

from gensim.models.tfidfmodel import TfidfModel
from gensim.models import Word2Vec
from nltk.corpus import brown
from gensim.models.keyedvectors import KeyedVectors
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from gensim.similarities import MatrixSimilarity
from scipy import spatial
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from scipy.stats import skew, kurtosis


def readData():
    train = pd.read_csv('./train_all_raw.csv')[:]
    test = pd.read_csv('./test_all_raw.csv')[:]
    return train, test


train , test = readData()

maxNumFeatures = 400

# bag of letter sequences (chars)
BagOfWordsExtractor = CountVectorizer(max_df=0.999, min_df=1000, max_features=maxNumFeatures, 
                                      analyzer='char', ngram_range=(1,3), 
                                      binary=True, lowercase=True)

BagOfWordsExtractor.fit(pd.concat([train.ix[:,'question1'],train.ix[:,'question2'],test.ix[:,'question1'],test.ix[:,'question2']]).unique())



def makeFeature(df_features):
	now = datetime.datetime.now()
	print now.strftime('%Y-%m-%d %H:%M:%S') 
	trainQuestion1_BOW_rep = BagOfWordsExtractor.transform(df_features.ix[:,'question1'])
	trainQuestion2_BOW_rep = BagOfWordsExtractor.transform(df_features.ix[:,'question2'])
	X = -(trainQuestion1_BOW_rep != trainQuestion2_BOW_rep).astype(int)
	df_features['f_bag_words'] = [X[i,:].toarray()[0] for i in range(0,len(df_features))]
	for j in range(0,len(df_features['f_bag_words'][0])):
		df_features['z_bag_words'+str(j)] = [df_features['f_bag_words'][i][j] for i in range(0,len(df_features))]
	df_features.fillna(0.0)
	now = datetime.datetime.now()
	print now.strftime('%Y-%m-%d %H:%M:%S') 
	return df_features




if __name__ == "__main__":
    train = makeFeature(train)
    col = [c for c in train.columns if c[:1]=='z']
    train.to_csv('train_bagofwords400.csv', index=False, columns = col)
    test = makeFeature(test)
    test.to_csv('test_bagofwords400.csv', index=False, columns = col)
    print("done bag of words")