#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import pandas as pd
import numpy as np
import nltk
import datetime
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.optimize import minimize
from gensim.utils import tokenize
from gensim.corpora.dictionary import Dictionary
stops = set(stopwords.words("english"))
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import multiprocessing
import difflib
import sys
import Levenshtein
import re
from string import punctuation

reload(sys)
sys.setdefaultencoding('utf-8')

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


def getdiffwords(q1, q2):
    word1 = q1.split()
    word2 = q2.split()
    qdf1 = [w for w in word1 if w not in word2]
    return " ".join(qdf1)

class MyCorpus(object):
    def __iter__(self):
        for x in tfidf_txt:
            yield dictionary.doc2bow(list(tokenize(x, errors='ignore')))

corpus = MyCorpus()
tfidf = TfidfModel(corpus)

def to_tfidf(text):
    res = tfidf[dictionary.doc2bow(list(tokenize(text, errors='ignore')))]
    return res

def tfidf_w(token):
    weights = dictionary.token2id
    if weights.has_key(token):
        res = tfidf.idfs[weights[token]]
    else:
        res = 1.0
    return res

def eucldist_vectorized(word_1, word_2):
    try:
        w2v1 = model[word_1]
        w2v2 = model[word_2]
        sim = np.sqrt(np.sum((np.array(w2v1) - np.array(w2v2))**2))
        return float(sim)
    except:
        return float(0)

# 输入两个wordlist
# 默认句子中每个词权重相同，实际可以更改
def getDiff(wordlist_1, wordlist_2):
    wordlist_1 = wordlist_1.split()
    wordlist_2 = wordlist_2.split()
    num = len(wordlist_1) + 0.001
    sim = 0.0
    for word_1 in wordlist_1:
        dis = 0.0
        for word_2 in wordlist_2:
            if (dis == 0.0):
                dis = eucldist_vectorized(word_1, word_2)
            else:
                dis = min(dis, eucldist_vectorized(word_1, word_2))
        sim += dis
    return (sim / num)


def getDiff_weight(wordlist_1, wordlist_2):
    wordlist_1 = wordlist_1.split()
    wordlist_2 = wordlist_2.split()
    tot_weights = 0.0
    for word_1 in wordlist_1:
        tot_weights += tfidf_w(word_1)
    sim = 0.0
    for word_1 in wordlist_1:
        dis = 0.0
        for word_2 in wordlist_2:
            if (dis == 0.0):
                dis = eucldist_vectorized(word_1, word_2)
            else:
                dis = min(dis, eucldist_vectorized(word_1, word_2))
        sim += tfidf_w(word_1) * dis
    return sim


def getDiff_averge(wordlist_1,wordlist_2):
    return getDiff_weight(wordlist_1,wordlist_2) + getDiff_weight(wordlist_2,wordlist_1)




def cos_sim(text1, text2):
    tfidf1 = to_tfidf(text1)
    tfidf2 = to_tfidf(text2)
    index = MatrixSimilarity([tfidf1],num_features=len(dictionary))
    sim = index[tfidf2]
    return float(sim[0])

#文本预处理
#print(dictionary)
def get_vector(text):
    # 建立一个全是0的array
    res =np.zeros([300])
    count = 0
    for word in word_tokenize(text):
        if word in vocab:
            res += tfidf_w(word) * model[word]
            count += tfidf_w(word)
    if count != 0:
        return res/count
    return  np.zeros([300])


def get_weight_vector(text):
    # 建立一个全是0的array
    res =np.zeros([300])
    count = 0
    for word in word_tokenize(text):
        if word in vocab:
            res += model[word]
            count += 1
    if count != 0:
        return res/count
    return  np.zeros([300])



def w2v_cos_sim(text1, text2):
    try:
        w2v1 = get_weight_vector(text1)
        w2v2 = get_weight_vector(text2)
        sim = 1 - spatial.distance.cosine(w2v1, w2v2)
        return float(sim)
    except:
        return float(0)

def get_features(df_features):
    print('use w2v to document presentation')
    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S')
   #df_features['z_document_dis'] = df_features.apply(lambda x: getDiff_averge(x['question1'], x['question2']), axis = 1)
    print('get_w2v')
    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S') 
    df_features['q1_unique'] = df_features.apply(lambda x: getdiffwords(x['question1'], x['question2']), axis = 1)
    df_features['q2_unique'] = df_features.apply(lambda x: getdiffwords(x['question2'], x['question1']), axis = 1)

    df_features['q1_unique_w2v_weight'] = df_features.q1_unique.map(lambda x: get_vector(" ".join(x)))
    df_features['q2_unique_w2v_weight'] = df_features.q2_unique.map(lambda x: get_vector(" ".join(x)))
    df_features['q1_unique_w2v'] = df_features.q1_unique.map(lambda x: get_weight_vector(" ".join(x)))
    df_features['q2_unique_w2v'] = df_features.q2_unique.map(lambda x: get_weight_vector(" ".join(x)))

    print('z_dist')
    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S') 
    #df_features['z_dist'] = df_features.apply(lambda x:Levenshtein.ratio(x['question1'], x['question2']), axis=1)
    now = datetime.datetime.now()
    print('z_tfidf_cos_sim')
    print now.strftime('%Y-%m-%d %H:%M:%S') 
    #df_features['z_tfidf_cos_sim'] = df_features.apply(lambda x: cos_sim(x['question1'], x['question2']), axis=1)
    now = datetime.datetime.now()
    print('z_w2v_calc')
    print now.strftime('%Y-%m-%d %H:%M:%S') 

    #df_features['z_w2v_unique'] = df_features.apply(lambda x: w2v_cos_sim(x['q1_unique'], x['q2_unique']), axis=1)
    df_features['z_w2v_unique_dis_e_weight'] = df_features.apply(lambda x: spatial.distance.euclidean(x['q1_unique_w2v_weight'], x['q2_unique_w2v_weight']), axis=1)
    df_features['z_w2v_unique_dis_e'] = df_features.apply(lambda x: spatial.distance.euclidean(x['q1_unique_w2v'], x['q2_unique_w2v']), axis=1)    
    
    df_features['z_w2v_unique_dis_mink_w'] = df_features.apply(lambda x: spatial.distance.minkowski(x['q1_unique_w2v_weight'], x['q2_unique_w2v_weight'],3), axis=1)
    df_features['z_w2v_unique_dis_cityblock_w'] = df_features.apply(lambda x: spatial.distance.cityblock(x['q1_unique_w2v_weight'], x['q2_unique_w2v_weight']), axis=1)
    df_features['z_w2v_unique_dis_canberra_w'] = df_features.apply(lambda x: spatial.distance.canberra(x['q1_unique_w2v_weight'], x['q2_unique_w2v_weight']), axis=1)
    
    df_features['z_w2v_unique_dis_mink'] = df_features.apply(lambda x: spatial.distance.minkowski(x['q1_unique_w2v'], x['q2_unique_w2v'],3), axis=1)
    df_features['z_w2v_unique_dis_cityblock'] = df_features.apply(lambda x: spatial.distance.cityblock(x['q1_unique_w2v'], x['q2_unique_w2v']), axis=1)
    df_features['z_w2v_unique_dis_canberra'] = df_features.apply(lambda x: spatial.distance.canberra(x['q1_unique_w2v'], x['q2_unique_w2v']), axis=1)
    
    df_features['z_q1_unique_skew_w'] = df_features.q1_unique_w2v_weight.map(lambda x:skew(x))
    df_features['z_q2_unique_skew_w'] = df_features.q2_unique_w2v_weight.map(lambda x:skew(x))
    df_features['z_q1_unique_kur_w'] = df_features.q1_unique_w2v_weight.map(lambda x:kurtosis(x))
    df_features['z_q2_unique_kur_w'] = df_features.q2_unique_w2v_weight.map(lambda x:kurtosis(x))


    df_features['z_q1_unique_skew'] = df_features.q1_unique_w2v.map(lambda x:skew(x))
    df_features['z_q2_unique_skew'] = df_features.q2_unique_w2v.map(lambda x:skew(x))
    df_features['z_q1_unique_kur'] = df_features.q1_unique_w2v.map(lambda x:kurtosis(x))
    df_features['z_q2_unique_kur'] = df_features.q2_unique_w2v.map(lambda x:kurtosis(x))
    del df_features['q1_unique_w2v_weight']
    del df_features['q2_unique_w2v_weight']
    del df_features['q1_unique_w2v']
    del df_features['q2_unique_w2v']
    print('all done')
    print now.strftime('%Y-%m-%d %H:%M:%S') 
    df_features.fillna(0.0)
    return df_features


if __name__ == '__main__':
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    vocab = model.vocab
    train = pd.read_csv('./train_all_raw.csv')[:]
    test = pd.read_csv('./test_all_raw.csv')[:]
    train = train.fillna('empty')
    test = test.fillna('empty')

    # clean
    tfidf_txt = train['question1'].tolist() + train['question2'].tolist() + test['question1'].tolist() + test[
        'question2'].tolist()
    train_qs = pd.Series(tfidf_txt).astype(str)
    dictionary = Dictionary(list(tokenize(x, errors='ignore')) for x in tfidf_txt)

    col = [c for c in train.columns if c[:1] == 'z' or c[:1] == 'f']

    train = get_features(train)
    train.to_csv('train_weight_noweight.csv', index=False, columns = col)

    test = get_features(test)
    test.to_csv('test_weight_noweight.csv', index=False,columns = col)



