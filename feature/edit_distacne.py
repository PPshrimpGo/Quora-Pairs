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

stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']

def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
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
    
    # Return a list of words
    return(text)



def str_stemmer(s):
    return " ".join([nltk.PorterStemmer().stem_word(word) for word in s.lower().split()])


model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
vocab = model.vocab
train = pd.read_csv('./input/train.csv')[:]
test = pd.read_csv('./input/test.csv')[:]
train = train.fillna('empty')
test = test.fillna('empty')

#clean

#train['question1'] =  train.question1.map(lambda x:text_to_wordlist(x))
#train['question2'] = train.question2.map(lambda x:text_to_wordlist(x))
#test['question1'] = test.question1.map(lambda x:text_to_wordlist(x))
#test['question2'] = test.question2.map(lambda x:text_to_wordlist(x))

tfidf_txt = train['question1'].tolist() + train['question2'].tolist() + test['question1'].tolist() + test['question2'].tolist()
dictionary = Dictionary(list(tokenize(x, errors='ignore')) for x in tfidf_txt)

class MyCorpus(object):
    def __iter__(self):
        for x in tfidf_txt:
            yield dictionary.doc2bow(list(tokenize(x, errors='ignore')))

corpus = MyCorpus()
tfidf = TfidfModel(corpus)

def to_tfidf(text):
    res = tfidf[dictionary.doc2bow(list(tokenize(text, errors='ignore')))]
    return res

def cos_sim(text1, text2):
    tfidf1 = to_tfidf(text1)
    tfidf2 = to_tfidf(text2)
    index = MatrixSimilarity([tfidf1],num_features=len(dictionary))
    sim = index[tfidf2]
    # 本来sim输出是一个array，我们不需要一个array来表示，
    # 所以我们直接cast成一个float
    return float(sim[0])

#文本预处理
#print(dictionary)
def get_vector(text):
    # 建立一个全是0的array
    res =np.zeros([300])
    count = 0
    for word in word_tokenize(text):
        if word in vocab:
            res += model[word]
            count += 1
    return res/count 

def w2v_cos_sim(text1, text2):
    try:
        w2v1 = get_vector(text1)
        w2v2 = get_vector(text2)
        sim = 1 - spatial.distance.cosine(w2v1, w2v2)
        return float(sim)
    except:
        return float(0)

def get_features(df_features):
    print('z_dist')
    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S') 
    df_features['z_dist'] = df_features.apply(lambda x:Levenshtein.ratio(x['question1'], x['question2']), axis=1)
    now = datetime.datetime.now()
    print('z_tfidf_cos_sim')
    print now.strftime('%Y-%m-%d %H:%M:%S') 
    df_features['z_tfidf_cos_sim'] = df_features.apply(lambda x: cos_sim(x['question1'], x['question2']), axis=1)
    now = datetime.datetime.now()
    print('z_w2v')
    print now.strftime('%Y-%m-%d %H:%M:%S') 
    df_features['z_w2v'] = df_features.apply(lambda x: w2v_cos_sim(x['question1'], x['question2']), axis=1)
    return df_features

train = get_features(train)
train.to_csv('train_featur_newf_notclean.csv', index=False)

col = [c for c in train.columns if c[:1]=='z']

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
params["eta"] = 0.02
params["subsample"] = 0.7
params["min_child_weight"] = 1
params["colsample_bytree"] = 0.7
params["max_depth"] = 4
params["silent"] = 1
params["seed"] = 1632

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=50, verbose_eval=100) #change to higher #s
print(log_loss(train.is_duplicate, bst.predict(xgb.DMatrix(train[col]))))

test = get_features(test)
test.to_csv('test_feature_newf_notclean.csv', index=False)

sub = pd.DataFrame()
sub['test_id'] = test['test_id']
sub['is_duplicate'] = bst.predict(xgb.DMatrix(test[col]))

sub.to_csv('submission_xgb_newf_notclean_02_04.csv', index=False)

