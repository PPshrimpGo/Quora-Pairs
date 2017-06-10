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
stops = set(stopwords.words("english"))
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import multiprocessing
import difflib
import re
from string import punctuation
import sys
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
    #text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"What's", "what is", text)
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

#word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
br = Word2Vec(brown.sents())

def get_similar(word):
    if word in br:
        lis = br.most_similar(word, topn=5)
        ret = []
        for one in lis:
            ret.append(one[0])
        return ret
    else:
        return [word]

train = pd.read_csv('./input/train.csv')[:]
test = pd.read_csv('./input/test.csv')[:]
train = train.fillna('empty')
test = test.fillna('empty')

train['question1'] =  train.question1.map(lambda x:text_to_wordlist(x))
train['question2'] = train.question2.map(lambda x:text_to_wordlist(x))
test['question1'] = test.question1.map(lambda x:text_to_wordlist(x))
test['question2'] = test.question2.map(lambda x:text_to_wordlist(x))


tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
#cvect = CountVectorizer(stop_words='english', ngram_range=(1, 1))

tfidf_txt = pd.Series(train['question1'].tolist() + train['question2'].tolist() + test['question1'].tolist() + test['question2'].tolist()).astype(str)
tfidf.fit_transform(tfidf_txt)
#cvect.fit_transform(tfidf_txt)

train_qs = pd.Series(train['question1'].tolist() + train['question2'].tolist() + test['question1'].tolist() + test['question2'].tolist()).astype(str)

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}


def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    if isinstance(st1, list):
        st1 = " ".join(st1)
        st2 = " ".join(st2)
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.ratio()

def word_match_share(q1words, q2words):
    if isinstance(q1words, str):
        q1words = q1words.split()
        q2words = q2words.split()
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words if w in q2words]
    shared_words_in_q2 = [w for w in q2words if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

def tfidf_word_match_share(word_1, word_2):
    if isinstance(word_1, str):
        word_1 = word_1.split()
        word_2 = word_2.split()
    shared_weights = [0] + [weights.get(w, 0) for w in word_1 if w in word_2] + [weights.get(w, 0) for w in word_2 if w in word_1]
    total_weights = [weights.get(w, 0) for w in word_1] + [weights.get(w, 0) for w in word_2]

    if (np.sum(shared_weights) == 0):
        return 0

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def expand_similar_word(qwords):
    qwords = qwords.split()
    word_1 = []
    for word in qwords:
        word_1.extend(get_similar(word))
    return " ".join(word_1)

def get_features(df_features):
    # now = datetime.datetime.now()
    # print now.strftime('%Y-%m-%d %H:%M:%S') 
    # print "matchnouns"
    # df_features['question1_nouns'] = df_features.question1.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    # df_features['question2_nouns'] = df_features.question2.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    # #df_features['z_noun_match'] = df_features.apply(lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)  #takes long
    # df_features['z_noun_match'] = df_features.apply(lambda r : tfidf_word_match_share(r.question1_nouns, r.question2_nouns), axis = 1)
    
    # now = datetime.datetime.now()
    # print now.strftime('%Y-%m-%d %H:%M:%S')   
    # print "matchverb"
    # df_features['question1_verbs'] = df_features.question1.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[0] == 'V' and t[1] == 'B'])
    # df_features['question2_verbs'] = df_features.question2.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[0] == 'V' and t[1] == 'B'])
    # #df_features['z_verb_match'] = df_features.apply(lambda r: sum([1 for w in r.question1_verbs if w in r.question2_verbs]), axis=1)  #takes long
    # df_features['z_verb_match'] = df_features.apply(lambda r : tfidf_word_match_share(r.question1_verbs, r.question2_verbs), axis = 1)
    
    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S') 
    # print "stem_tfidf"
    # df_features['q1_stem'] = df_features.question1.map(lambda x: [w for w in nltk.PorterStemmer().stem_word(str(x).lower()).split(' ')])
    # df_features['q2_stem'] = df_features.question2.map(lambda x: [w for w in nltk.PorterStemmer().stem_word(str(x).lower()).split(' ')])
    # #df_features['z_adj_match'] = df_features.apply(lambda r: sum([1 for w in r.question1_adjs if w in r.question2_adjs]), axis=1)  #takes long
    # df_features['z_stem_tfidf'] = df_features.apply(lambda r : tfidf_word_match_share(r.q1_stem, r.q2_stem), axis = 1)
    # now = datetime.datetime.now()
    # df_features['q1_expand'] = df_features.question1.map(lambda x: expand_similar_word(x))
    # df_features['q2_expand'] = df_features.question2.map(lambda x: expand_similar_word(x))
    # df_features['question1_nouns'] = df_features.question1.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    # df_features['question2_nouns'] = df_features.question2.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])

    print('tf_idf_share...')
    df_features['z_raw_tfidf'] = df_features.apply(lambda r : tfidf_word_match_share(r.question1, r.question2), axis = 1)
    df_features['z_none_tfidf'] = df_features.apply(lambda r : tfidf_word_match_share(r.question1_nouns, r.question2_nouns), axis = 1)
    df_features['z_expand_tfidf'] = df_features.apply(lambda r : tfidf_word_match_share(r.q1_expand, r.q2_expand), axis = 1)
    df_features['z_adjs_tfidf'] = df_features.apply(lambda r : tfidf_word_match_share(r.question1_adjs, r.question2_adjs), axis = 1)
    df_features['z_verbs_tfidf'] = df_features.apply(lambda r : tfidf_word_match_share(r.question1_verbs, r.question2_verbs), axis = 1)

    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S') 
    print('match...')
    df_features['z_raw_match'] = df_features.apply(lambda r: sum([1 for w in r.question1.split()  if w in r.question2.split()]), axis=1)  #takes long
    df_features['z_noun_match'] = df_features.apply(lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)  #takes long
    df_features['z_expand_match'] = df_features.apply(lambda r: sum([1 for w in r.q1_expand.split() if w in r.q2_expand.split()]), axis=1)  #takes long
    
    print('lengths...')
    df_features['z_len1'] = df_features.question1.map(lambda x: len(str(x)))
    df_features['z_len2'] = df_features.question2.map(lambda x: len(str(x)))
    df_features['z_word_len1'] = df_features.question1.map(lambda x: len(str(x).split()))
    df_features['z_word_len2'] = df_features.question2.map(lambda x: len(str(x).split()))
    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S') 

    print('difflib...')
    df_features['z_raw_diff_ratio'] = df_features.apply(lambda r: diff_ratios(r.question1, r.question2), axis=1)  #takes long
    df_features['z_expand_diff_ratio'] = df_features.apply(lambda r: diff_ratios(r.q1_expand, r.q2_expand), axis=1)  #takes long
    df_features['z_none_diff_ratio'] = df_features.apply(lambda r: diff_ratios(r.question1_nouns, r.question2_nouns), axis=1)  #takes long

    print('word match...')
    df_features['z_word_match'] = df_features.apply(lambda r: word_match_share(r.question1, r.question2), axis=1, raw=True)
    df_features['z_expand_match'] = df_features.apply(lambda r: word_match_share(r.q1_expand, r.q2_expand), axis=1, raw=True)
    df_features['z_none_match'] = df_features.apply(lambda r: word_match_share(r.question1_nouns, r.question2_nouns), axis=1, raw=True)

    print('tfidf...')
    df_features['z_tfidf_sum1'] = df_features.question1.map(lambda x: np.sum(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_sum2'] = df_features.question2.map(lambda x: np.sum(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_mean1'] = df_features.question1.map(lambda x: np.mean(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_mean2'] = df_features.question2.map(lambda x: np.mean(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_len1'] = df_features.question1.map(lambda x: len(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_len2'] = df_features.question2.map(lambda x: len(tfidf.transform([str(x)]).data))
    return df_features.fillna(0.0)


train = get_features(train)
train.to_csv('train_ttt.csv', index=False)

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
test.to_csv('test_ttt.csv', index=False)

sub = pd.DataFrame()
sub['test_id'] = test['test_id']
sub['is_duplicate'] = bst.predict(xgb.DMatrix(test[col]))

sub.to_csv('submission_ttt.csv', index=False)
