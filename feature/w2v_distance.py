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

stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']

def text_to_wordlist(text, remove_stop_words=True, stem_words=False, lemma=True):
    # Clean the text, with the option to remove stop_words and to stem words.

    # Clean the text
    text = text.rstrip('?')
    text = text.rstrip(',')
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        #stemmer = SnowballStemmer('english')
        #stemmed_words = [stemmer.stem(word) for word in text]
        stemmed_words = [nltk.PorterStemmer().stem_word(word.lower()) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words after lemma
    if lemma:
        text = text.split()
        lancaster_stemmer = LancasterStemmer()
        lemma_words = [lancaster_stemmer.stem(word.lower()) for word in text]
        text = " ".join(lemma_words)
    return(text)



def str_stemmer(s):
    return " ".join([nltk.PorterStemmer().stem_word(word) for word in s.lower().split()])

def getdiffwords(q1, q2):
    word1 = q1.split()
    word2 = q2.split()
    qdf1 = [w for w in word1 if w not in word2]
    return " ".join(qdf1)

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
vocab = model.vocab
train = pd.read_csv('./train_all_raw.csv')[:]
test = pd.read_csv('./test_all_raw.csv')[:]
train = train.fillna('empty')
test = test.fillna('empty')

#clean

#train['question1'] =  train.question1.map(lambda x:text_to_wordlist(x))
#train['question2'] = train.question2.map(lambda x:text_to_wordlist(x))
#test['question1'] = test.question1.map(lambda x:text_to_wordlist(x))
#test['question2'] = test.question2.map(lambda x:text_to_wordlist(x))
tfidf_txt = train['question1'].tolist() + train['question2'].tolist() + test['question1'].tolist() + test['question2'].tolist()

train_qs = pd.Series(tfidf_txt).astype(str)

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

dictionary = Dictionary(list(tokenize(x, errors='ignore')) for x in tfidf_txt)

class MyCorpus(object):
    def __iter__(self):
        for x in tfidf_txt:
            yield dictionary.doc2bow(list(tokenize(x, errors='ignore')))

corpus = MyCorpus()
tfidf = TfidfModel(corpus)

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
        tot_weights += weights[word_1]
    sim = 0.0
    for word_1 in wordlist_1:
        dis = 0.0
        for word_2 in wordlist_2:
            if (dis == 0.0):
                dis = eucldist_vectorized(word_1, word_2)
            else:
                dis = min(dis, eucldist_vectorized(word_1, word_2))
        sim += weights[word_1] * dis
    return sim

def getDiff_weight_tfidf(wordlist_1, wordlist_2):
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


def getDiff_averge_tfidf(wordlist_1,wordlist_2):
    return getDiff_weight_tfidf(wordlist_1,wordlist_2) + getDiff_weight_tfidf(wordlist_2,wordlist_1)



def to_tfidf(text):
    res = tfidf[dictionary.doc2bow(list(tokenize(text, errors='ignore')))]
    return res

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
            res += weights[word] * model[word]
            count += weights[word]
    if count != 0:
        return res/count
    return  np.zeros([300])


def get_vector_tfidf(text):
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

def w2v_cos_sim_tfidf(text1, text2):
    try:
        w2v1 = get_vector_tfidf(text1)
        w2v2 = get_vector_tfidf(text2)
        sim = 1 - spatial.distance.cosine(w2v1, w2v2)
        return float(sim)
    except:
        return float(0)

def get_features(df_features):
    print('use w2v to document presentation')
    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S')
    df_features['z_document_dis'] = df_features.apply(lambda x: getDiff_averge_tfidf(x['question1'], x['question2']), axis = 1)
    print('nones')
    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S') 
    df_features['q1_unique'] = df_features.apply(lambda x: getdiffwords(x['question1'], x['question2']), axis = 1)
    df_features['q2_unique'] = df_features.apply(lambda x: getdiffwords(x['question2'], x['question1']), axis = 1)
    #df_features['question1_nouns'] = df_features.question1.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    #df_features['question2_nouns'] = df_features.question2.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    df_features['question1_w2v'] = df_features.question1.map(lambda x: get_vector_tfidf(" ".join(x)))
    df_features['question2_w2v'] = df_features.question2.map(lambda x: get_vector_tfidf(" ".join(x)))
    print('z_dist')
    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S') 
    df_features['z_dist'] = df_features.apply(lambda x:Levenshtein.ratio(x['question1'], x['question2']), axis=1)
    now = datetime.datetime.now()
    print('z_tfidf_cos_sim')
    print now.strftime('%Y-%m-%d %H:%M:%S') 
    df_features['z_tfidf_cos_sim'] = df_features.apply(lambda x: cos_sim(x['question1'], x['question2']), axis=1)
    now = datetime.datetime.now()
    print('z_w2v_nones')
    print now.strftime('%Y-%m-%d %H:%M:%S') 
    df_features['z_w2v_unique'] = df_features.apply(lambda x: w2v_cos_sim_tfidf(x['q1_unique'], x['q2_unique']), axis=1)
    df_features['z_w2v_dis_e'] = df_features.apply(lambda x: spatial.distance.euclidean(x['question1_w2v'], x['question2_w2v']), axis=1)
    df_features['z_w2v_dis_mink'] = df_features.apply(lambda x: spatial.distance.minkowski(x['question1_w2v'], x['question2_w2v'],3), axis=1)
    df_features['z_w2v_dis_cityblock'] = df_features.apply(lambda x: spatial.distance.cityblock(x['question1_w2v'], x['question2_w2v']), axis=1)
    df_features['z_w2v_dis_canberra'] = df_features.apply(lambda x: spatial.distance.canberra(x['question1_w2v'], x['question2_w2v']), axis=1)
    df_features['z_q1_skew'] = df_features.question1_w2v.map(lambda x:skew(x))
    df_features['z_q2_skew'] = df_features.question2_w2v.map(lambda x:skew(x))
    df_features['z_q1_kur'] = df_features.question1_w2v.map(lambda x:kurtosis(x))
    df_features['z_q2_kur'] = df_features.question2_w2v.map(lambda x:kurtosis(x))
    del df_features['question1_w2v']
    del df_features['question2_w2v']
    print('all done')
    print now.strftime('%Y-%m-%d %H:%M:%S') 
    df_features.fillna(0.0)
    return df_features


if __name__ == '__main__':
    train = get_features(train)
    train.to_csv('train_weight_tfidf.csv', index=False)

    test = get_features(test)
    test.to_csv('test_weight_tfidf.csv', index=False)

