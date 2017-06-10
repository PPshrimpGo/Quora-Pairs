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

def weighted_Overlap(q1words, q2words, weights):
    if isinstance(q1words, str):
        q1words = q1words.split()
        q2words = q2words.split()
    tot = 0.0
    for word in q1words:
        if word in weights:
            tot += math.exp(weights[word])
    for word in q2words:
        if word in weights:
            tot += math.exp(weights[word])
    ret = 0.0
    for word1 in q1words:
        for word2 in q2words:
            if (word1 == word2 and word1 in weights):
                ret += 2 * math.exp(weights[word1]) / (tot+0.000001)
    return ret


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

def Cosine(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    Lx = np.sqrt(vec1.dot(vec1))
    Ly = np.sqrt(vec2.dot(vec2))
    return vec1.dot(vec2) / ((Lx * Ly)+0.000001)

def Manhatton(vec1, vec2):
    return np.sum(np.fabs(np.array(vec1) - np.array(vec2)))

def Euclidean(vec1, vec2):
    return np.sqrt(np.sum(np.array(vec1) - np.array(vec2)) ** 2)

def PearsonSimilar(vec1, vec2):
    data = np.vstack((vec1, vec2))
    return pd.DataFrame(data).T.corr('pearson')[0][1]

def SpearmanSimilar(vec1, vec2):
    data = np.vstack((vec1, vec2))
    return pd.DataFrame(data).T.corr('spearman')[0][1]

def KendallSimilar(vec1, vec2):
    data = np.vstack((vec1, vec2))
    return pd.DataFrame(data).T.corr('kendall')[0][1]

def getVec(sentence, model):
    if isinstance(sentence, str):
        sentence = sentence.split()
    ret = np.zeros([300])
    # ret = np.zeros([100]) # brwon 100 dim
    count = 0
    for word in sentence:
        if word in model:
            ret += model[word]
            count += 1
    if count != 0:
        return (ret / count)
    # return np.zeros([100])
    return np.zeros([300])

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

def word_weights(data):
    train_qs = pd.Series(data['q1_expand'].tolist() + data['q2_expand'].tolist())
    eps = 5000
    words = (" ".join(train_qs)).lower().split()
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
    return weights

# 输入两个wordlist
# 默认句子中每个词权重相同，实际可以更改

train, test = readData()
train_qs = pd.Series(train['q1_expand'].tolist() + train['q2_expand'].tolist() + test['q1_expand'].tolist() + test['q2_expand'].tolist()).astype(str)
eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

def getDiff(wordlist_1, wordlist_2, discal, model):
    sim = 0.0
    for word_1 in wordlist_1:
        dis = 0.0
        for word_2 in wordlist_2:
            if word_1 not in model:
                continue
            if word_2 not in model:
                continue
            if (dis == 0.0):
                dis = discal(model[word_1], model[word_2])
            else:
                dis = min(dis, discal(model[word_1], model[word_2]))
        sim += weights[word_1.lower()] * dis
    return sim

def getfromw2v(wordlist_1, wordlist_2, discal, model):
    if isinstance(wordlist_1, str):
        wordlist_1 = wordlist_1.split()
        wordlist_2 = wordlist_2.split()
    return getDiff(wordlist_1, wordlist_2, discal, model) + getDiff(wordlist_2, wordlist_1, discal, model)


def makeFeature(df_features):
    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S')
    print ('get sentence vector')
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # model = KeyedVectors.load_word2vec_format('glove.6B.300d.txt', binary=False)
    # model = Word2Vec(brown.sents())
    df_features['vec1'] = df_features.q1_expand.map(lambda x: getVec(x, model))
    df_features['vec2'] = df_features.q2_expand.map(lambda x: getVec(x, model))

    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S')
    print ('get six kinds of coefficient about vector')
    df_features['f_cosine'] = df_features.apply(lambda x: Cosine(x['vec1'], x['vec2']), axis=1)
    df_features['f_manhatton'] = df_features.apply(lambda x: Manhatton(x['vec1'], x['vec2']), axis=1)
    df_features['f_euclidean'] = df_features.apply(lambda x: Euclidean(x['vec1'], x['vec2']), axis=1)
    df_features['f_pearson'] = df_features.apply(lambda x: PearsonSimilar(x['vec1'], x['vec2']), axis=1)
    df_features['f_spearman'] = df_features.apply(lambda x: SpearmanSimilar(x['vec1'], x['vec2']), axis=1)
    df_features['f_kendall'] = df_features.apply(lambda x: KendallSimilar(x['vec1'], x['vec2']), axis=1)

    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S')
    print ('get 3 kinds of coefficient about from w2c 2 document')
    df_features['f_cosine_w2v'] = df_features.apply(lambda x: getfromw2v(x['q1_expand'], x['q2_expand'],Cosine, model), axis=1)
    df_features['f_euclidean_w2v'] = df_features.apply(lambda x: getfromw2v(x['q1_expand'], x['q2_expand'],Euclidean, model), axis=1)
    df_features['f_manhatton_w2v'] = df_features.apply(lambda x: getfromw2v(x['q1_expand'], x['q2_expand'],Manhatton, model), axis=1)

    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S')
    print ('get three kinds of coefficient about nouns, verb, adj')
    df_features['f_raw_jaccarc'] = df_features.apply(lambda x: Jaccarc(x['question1'], x['question2']), axis=1)
    df_features['f_raw_dice'] = df_features.apply(lambda x: Dice(x['question1'], x['question2']),axis=1)
    df_features['f_raw_ochiai'] = df_features.apply(lambda x: Ochiai(x['question1'], x['question2']), axis=1)
    df_features['f_expand_jaccarc'] = df_features.apply(lambda x: Jaccarc(x['q1_expand'], x['q2_expand']), axis=1)
    df_features['f_expand_dice'] = df_features.apply(lambda x: Dice(x['q1_expand'], x['q2_expand']),axis=1)
    df_features['f_expand_ochiai'] = df_features.apply(lambda x: Ochiai(x['q1_expand'], x['q2_expand']), axis=1)
    df_features['f_nouns_jaccarc'] = df_features.apply(lambda x: Jaccarc(x['question1_nouns'], x['question2_nouns']), axis=1)
    df_features['f_nouns_dice'] = df_features.apply(lambda x: Dice(x['question1_nouns'], x['question2_nouns']),axis=1)
    df_features['f_nouns_ochiai'] = df_features.apply(lambda x: Ochiai(x['question1_nouns'], x['question2_nouns']), axis=1)
    df_features['f_verbs_jaccarc'] = df_features.apply(lambda x: Jaccarc(x['question1_verbs'], x['question2_verbs']), axis=1)
    df_features['f_verbs_dice'] = df_features.apply(lambda x: Dice(x['question1_verbs'], x['question2_verbs']),axis=1)
    df_features['f_verbs_ochiai'] = df_features.apply(lambda x: Ochiai(x['question1_verbs'], x['question2_verbs']), axis=1)
    df_features['f_adjs_jaccarc'] = df_features.apply(lambda x: Jaccarc(x['question1_adjs'], x['question2_adjs']), axis=1)
    df_features['f_adjs_dice'] = df_features.apply(lambda x: Dice(x['question1_adjs'], x['question2_adjs']),axis=1)
    df_features['f_adjs_ochiai'] = df_features.apply(lambda x: Ochiai(x['question1_adjs'], x['question2_adjs']), axis=1)

    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S')
    print ('get weighted overlap about expand')
    weights = word_weights(df_features)
    df_features['f_weighted_overlap'] = df_features.apply(lambda x: weighted_Overlap(x['q1_expand'], x['q2_expand'], weights), axis=1)

    print('all done')
    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S')
    df_features.fillna(0.0)
    return df_features

if __name__ == "__main__":

    train = makeFeature(train)
    train.to_csv('train_string_based_feature.csv', index=False)
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
    test.to_csv('test_string_based_feature.csv', index=False)
    #test = [c for c in test_features.columns if c[:1] == 'f']
    sub = pd.DataFrame()
    sub['test_id'] = test['test_id']
    sub['is_duplicate'] = bst.predict(xgb.DMatrix(test[col]))

    sub.to_csv('summit_stringbase.csv', index=False)
