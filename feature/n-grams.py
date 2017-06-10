#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import pandas as pd
import numpy as np
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


from nltk import ngrams
from simhash import Simhash
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

from multiprocessing import Pool #We use pool to speed up feature creation

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)



def tokenize(sequence):
    #words = word_tokenize(sequence)
    words = sequence.split(' ')
    #filtered_words = [word for word in words if word not in stopwords.words('english')]
    return words

def readData():
    train = pd.read_csv('./train_all_raw.csv', dtype={'question1': str, 'question2': str},encoding='utf-8')[:]
    test = pd.read_csv('./test_all_raw.csv', dtype={'question1': str, 'question2': str},encoding='utf-8')[:]
    return train, test


def Jaccarc(q1words, q2words):
    if isinstance(q1words, str):
        q1words = q1words.split()
        q2words = q2words.split()
    tot = len(q1words) + len(q2words) + 1
    same = 0
    for word1 in q1words:
        for word2 in q2words:
            if (word1 == word2):
                same += 1
    return float(same) / (float(tot - same)+0.000001)


def Dice(q1words, q2words):
    if isinstance(q1words, str):
        q1words = q1words.split()
        q2words = q2words.split()
    tot = len(q1words) + len(q2words) + 1
    same = 0
    for word1 in q1words:
        for word2 in q2words:
            if (word1 == word2):
                same += 1
    return 2 * float(same) / (float(tot)+0.000001)

def Ochiai(q1words, q2words):
    if isinstance(q1words, str):
        q1words = q1words.split()
        q2words = q2words.split()
    tot = len(q1words) * len(q2words) + 1
    same = 0
    for word1 in q1words:
        for word2 in q2words:
            if (word1 == word2):
                same += 1
    return float(same) / (np.sqrt(float(tot))+0.000001)

#n-grams 
def get_word_ngrams(sequence, n=3):
    tokens = tokenize(sequence)
    return [' '.join(ngram) for ngram in ngrams(tokens, n)]


def get_character_ngrams(sequence, n=3):
    return [sequence[i:i+n] for i in range(len(sequence)-n+1)]

def caluclate_simhash_distance(sequence1, sequence2):
    return Simhash(sequence1).distance(Simhash(sequence2))


def get_word_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = tokenize(q1), tokenize(q2)
    return caluclate_simhash_distance(q1, q2)

def get_word_2gram_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return caluclate_simhash_distance(q1, q2)

def get_char_2gram_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
    return caluclate_simhash_distance(q1, q2)

def get_word_3gram_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return caluclate_simhash_distance(q1, q2)

def get_char_3gram_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
    return caluclate_simhash_distance(q1, q2)



def get_word_distance2(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = tokenize(q1), tokenize(q2)
    return Jaccarc(q1, q2)

def get_word_2gram_distance2(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return Jaccarc(q1, q2)

def get_char_2gram_distance2(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
    return Jaccarc(q1, q2)

def get_word_3gram_distance2(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return Jaccarc(q1, q2)

def get_char_3gram_distance2(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
    return Jaccarc(q1, q2)




def get_word_distance3(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = tokenize(q1), tokenize(q2)
    return Dice(q1, q2)

def get_word_2gram_distance3(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return Dice(q1, q2)

def get_char_2gram_distance3(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
    return Dice(q1, q2)

def get_word_3gram_distance3(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return Dice(q1, q2)

def get_char_3gram_distance3(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
    return Dice(q1, q2)




def get_word_distance4(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = tokenize(q1), tokenize(q2)
    return Ochiai(q1, q2)

def get_word_2gram_distance4(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return Ochiai(q1, q2)

def get_char_2gram_distance4(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
    return Ochiai(q1, q2)

def get_word_3gram_distance4(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return Ochiai(q1, q2)

def get_char_3gram_distance4(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
    return Ochiai(q1, q2)


train, test = readData()
train['questions'] = train['question1'] + '_split_tag_' + train['question2']
test['questions'] = test['question1'] + '_split_tag_' + test['question2']
train['questions_expand'] = train['q1_expand'] + '_split_tag_' + train['q2_expand']
test['questions_expand'] = test['q1_expand'] + '_split_tag_' + test['q2_expand']

def makeFeature(df_features):
    pool = Pool(processes=20)
    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S')
    print ('get n-grams')
    df_features['f_1dis'] = pool.map(get_word_distance, df_features['questions'])
    df_features['f_2word_dis'] = pool.map(get_word_2gram_distance, df_features['questions'])
    df_features['f_2char_dis'] = pool.map(get_char_2gram_distance, df_features['questions'])
    df_features['f_3word_dis'] = pool.map(get_word_3gram_distance, df_features['questions'])
    df_features['f_3char_dis'] = pool.map(get_char_3gram_distance, df_features['questions'])


    df_features['f_1dis2'] = pool.map(get_word_distance2, df_features['questions'])
    df_features['f_2word_dis2'] = pool.map(get_word_2gram_distance2, df_features['questions'])
    df_features['f_2char_dis2'] = pool.map(get_char_2gram_distance2, df_features['questions'])
    df_features['f_3word_dis2'] = pool.map(get_word_3gram_distance2, df_features['questions'])
    df_features['f_3char_dis2'] = pool.map(get_char_3gram_distance2, df_features['questions'])



    df_features['f_1dis3'] = pool.map(get_word_distance3, df_features['questions'])
    df_features['f_2word_dis3'] = pool.map(get_word_2gram_distance3, df_features['questions'])
    df_features['f_2char_dis3'] = pool.map(get_char_2gram_distance3, df_features['questions'])
    df_features['f_3word_dis3'] = pool.map(get_word_3gram_distance3, df_features['questions'])
    df_features['f_3char_dis3'] = pool.map(get_char_3gram_distance3, df_features['questions'])



    df_features['f_1dis4'] = pool.map(get_word_distance4, df_features['questions'])
    df_features['f_2word_dis4'] = pool.map(get_word_2gram_distance4, df_features['questions'])
    df_features['f_2char_dis4'] = pool.map(get_char_2gram_distance4, df_features['questions'])
    df_features['f_3word_dis4'] = pool.map(get_word_3gram_distance4, df_features['questions'])
    df_features['f_3char_dis4'] = pool.map(get_char_3gram_distance4, df_features['questions'])


    ### expand

    df_features['f_1dise'] = pool.map(get_word_distance, df_features['questions_expand'])
    df_features['f_2word_dise'] = pool.map(get_word_2gram_distance, df_features['questions_expand'])
    df_features['f_2char_dise'] = pool.map(get_char_2gram_distance, df_features['questions_expand'])
    df_features['f_3word_dise'] = pool.map(get_word_3gram_distance, df_features['questions_expand'])
    df_features['f_3char_dise'] = pool.map(get_char_3gram_distance, df_features['questions_expand'])


    df_features['f_1dis2e'] = pool.map(get_word_distance2, df_features['questions_expand'])
    df_features['f_2word_dis2e'] = pool.map(get_word_2gram_distance2, df_features['questions_expand'])
    df_features['f_2char_dis2e'] = pool.map(get_char_2gram_distance2, df_features['questions_expand'])
    df_features['f_3word_dis2e'] = pool.map(get_word_3gram_distance2, df_features['questions_expand'])
    df_features['f_3char_dis2e'] = pool.map(get_char_3gram_distance2, df_features['questions_expand'])



    df_features['f_1dis3e'] = pool.map(get_word_distance3, df_features['questions_expand'])
    df_features['f_2word_dis3e'] = pool.map(get_word_2gram_distance3, df_features['questions_expand'])
    df_features['f_2char_dis3e'] = pool.map(get_char_2gram_distance3, df_features['questions_expand'])
    df_features['f_3word_dis3e'] = pool.map(get_word_3gram_distance3, df_features['questions_expand'])
    df_features['f_3char_dis3e'] = pool.map(get_char_3gram_distance3, df_features['questions_expand'])



    df_features['f_1dis4e'] = pool.map(get_word_distance4, df_features['questions_expand'])
    df_features['f_2word_dis4e'] = pool.map(get_word_2gram_distance4, df_features['questions_expand'])
    df_features['f_2char_dis4e'] = pool.map(get_char_2gram_distance4, df_features['questions_expand'])
    df_features['f_3word_dis4e'] = pool.map(get_word_3gram_distance4, df_features['questions_expand'])
    df_features['f_3char_dis4e'] = pool.map(get_char_3gram_distance4, df_features['questions_expand'])

    print('all done')
    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S')
    df_features.fillna(0.0)
    return df_features

if __name__ == "__main__":

    train = makeFeature(train)
    train.to_csv('train_gram_feature.csv', index=False)
    #train = [c for c in train.columns if c[:1] == 'f']
    #############
    col = [c for c in train.columns if c[:1]=='f']
    pos_train = train[train['is_duplicate'] == 1]
    neg_train = train[train['is_duplicate'] == 0]
    p = 0.165
    scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    while scale > 1:
        neg_train = pd.concat([neg_train, neg_train])
        scale -=1
    neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    train = pd.concat([pos_train, neg_train])

    x_train, x_valid, y_train, y_valid = train_test_split(train[col], train['is_duplicate'], test_size=0.2, random_state=0)

    params = {}
    params["objective"] = "binary:logistic"
    params['eval_metric'] = 'logloss'
    params["eta"] = 0.04
    params["subsample"] = 0.9
    params["min_child_weight"] = 1
    params["colsample_bytree"] = 0.8
    params["max_depth"] = 8
    params["silent"] = 1
    params["seed"] = 1632
    params["gama"] = 0.005

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    bst = xgb.train(params, d_train, 600, watchlist, early_stopping_rounds=50, verbose_eval=100) #change to higher #s
    print(log_loss(train.is_duplicate, bst.predict(xgb.DMatrix(train[col]))))

    #############

    test = makeFeature(test)
    test.to_csv('test_gram_feature.csv', index=False)
    #test = [c for c in test_features.columns if c[:1] == 'f']
    sub = pd.DataFrame()
    sub['test_id'] = test['test_id']
    sub['is_duplicate'] = bst.predict(xgb.DMatrix(test[col]))

    sub.to_csv('summit_gram.csv', index=False)
