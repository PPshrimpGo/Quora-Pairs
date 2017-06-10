from __future__ import division

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm, tqdm_notebook
import itertools as it
import pickle
import glob
import os
import string

from scipy import sparse

import nltk
import spacy

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, make_scorer
from sklearn.decomposition import TruncatedSVD

from scipy.optimize import minimize

import eli5

import xgboost as xgb

# read data
df_train = pd.read_csv('./input/train.csv', 
                       dtype={
                           'question1': np.str,
                           'question2': np.str
                       })[:]
df_train['test_id'] = -1
df_test = pd.read_csv('./input/test.csv', 
                      dtype={
                          'question1': np.str,
                          'question2': np.str
                      })[:]
df_test['id'] = -1
df_test['qid1'] = -1
df_test['qid2'] = -1
df_test['is_duplicate'] = -1

df = pd.concat([df_train, df_test])
df['question1'] = df['question1'].fillna('')
df['question2'] = df['question2'].fillna('')
df['uid'] = np.arange(df.shape[0])
df = df.set_index(['uid'])
del(df_train, df_test)

ix_train = np.where(df['id'] >= 0)[0]
ix_test = np.where(df['id'] == -1)[0]
ix_is_dup = np.where(df['is_duplicate'] == 1)[0]
ix_not_dup = np.where(df['is_duplicate'] == 0)[0]

# Letter n-gram
if os.path.isfile('./cv_char.pkl') and os.path.isfile('./ch_freq.pkl'):
    with open('./cv_char.pkl', 'rb') as f:
        cv_char = pickle.load(f)
    with open('./ch_freq.pkl', 'rb') as f:
        ch_freq = pickle.load(f)
else:
    cv_char = CountVectorizer(ngram_range=(1, 3), analyzer='char')
    ch_freq = np.array(cv_char.fit_transform(df['question1'].tolist() + df['question2'].tolist()).sum(axis=0))[0, :]
    with open('./cv_char.pkl', 'wb') as f:
        pickle.dump(cv_char, f)
    with open('./ch_freq.pkl', 'wb') as f:
        pickle.dump(ch_freq, f)

# get unigrams, bigrams, trigrams
unigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 1])
ix_unigrams = np.sort(unigrams.values())
print 'Unigrams:', len(unigrams)
bigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 2])
ix_bigrams = np.sort(bigrams.values())
print 'Bigrams: ', len(bigrams)
trigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 3])
ix_trigrams = np.sort(trigrams.values())
print 'Trigrams:', len(trigrams)

# get m_q1 && m_q2
def save_sparse_csr(fname, sm):
    np.savez(fname, 
             data=sm.data, 
             indices=sm.indices,
             indptr=sm.indptr, 
             shape=sm.shape)

def load_sparse_csr(fname):
    loader = np.load(fname)
    return sparse.csr_matrix((
        loader['data'], 
        loader['indices'], 
        loader['indptr']),
        shape=loader['shape'])

if os.path.isfile('./m_q1.npz') and os.path.isfile('./m_q2.npz'):
    m_q1 = load_sparse_csr('./m_q1.npz')
    m_q2 = load_sparse_csr('./m_q2.npz')
else:
    m_q1 = cv_char.transform(df['question1'].values)
    m_q2 = cv_char.transform(df['question2'].values)
    save_sparse_csr('./m_q1.npz', m_q1)
    save_sparse_csr('./m_q2.npz', m_q2)
"""
this is all about unigram featutes
"""
# unigram_jaccard
v_num = (m_q1[:, ix_unigrams] > 0).minimum((m_q2[:, ix_unigrams] > 0)).sum(axis=1)
v_den = (m_q1[:, ix_unigrams] > 0).maximum((m_q2[:, ix_unigrams] > 0)).sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_unigram_jaccard'] = v_score

# unigram_all_jaccard
v_num = m_q1[:, ix_unigrams].minimum(m_q2[:, ix_unigrams]).sum(axis=1)
v_den = m_q1[:, ix_unigrams].sum(axis=1) + m_q2[:, ix_unigrams].sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_unigram_all_jaccard'] = v_score

# unigram_all_jaccard_max
v_num = m_q1[:, ix_unigrams].minimum(m_q2[:, ix_unigrams]).sum(axis=1)
v_den = m_q1[:, ix_unigrams].maximum(m_q2[:, ix_unigrams]).sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_unigram_all_jaccard_max'] = v_score


"""
This is all about bigram features 
"""
# bigram_jaccard
v_num = (m_q1[:, ix_bigrams] > 0).minimum((m_q2[:, ix_bigrams] > 0)).sum(axis=1)
v_den = (m_q1[:, ix_bigrams] > 0).maximum((m_q2[:, ix_bigrams] > 0)).sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_bigram_jaccard'] = v_score

# bigram_all_jaccard
v_num = m_q1[:, ix_bigrams].minimum(m_q2[:, ix_bigrams]).sum(axis=1)
v_den = m_q1[:, ix_bigrams].sum(axis=1) + m_q2[:, ix_bigrams].sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_bigram_all_jaccard'] = v_score

# bigram_all_jaccard_max
v_num = m_q1[:, ix_bigrams].minimum(m_q2[:, ix_bigrams]).sum(axis=1)
v_den = m_q1[:, ix_bigrams].maximum(m_q2[:, ix_bigrams]).sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_bigram_all_jaccard_max'] = v_score


"""
this is all about trigrams features
"""
m_q1 = m_q1[:, ix_trigrams]
m_q2 = m_q2[:, ix_trigrams]

# trigram_jaccard
v_num = (m_q1 > 0).minimum((m_q2 > 0)).sum(axis=1)
v_den = (m_q1 > 0).maximum((m_q2 > 0)).sum(axis=1)
v_den[np.where(v_den == 0)] = 1
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_trigram_jaccard'] = v_score

# trigram_all_jaccard
v_num = m_q1.minimum(m_q2).sum(axis=1)
v_den = m_q1.sum(axis=1) + m_q2.sum(axis=1)
v_den[np.where(v_den == 0)] = 1
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_trigram_all_jaccard'] = v_score

# trigram_all_jaccard_max
v_num = m_q1.minimum(m_q2).sum(axis=1)
v_den = m_q1.maximum(m_q2).sum(axis=1)
v_den[np.where(v_den == 0)] = 1
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_trigram_all_jaccard_max'] = v_score

# trigram_tfidf_cosine
tft = TfidfTransformer(
    norm='l2', 
    use_idf=True, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_num = np.array(m_q1_tf.multiply(m_q2_tf).sum(axis=1))[:, 0]
v_den = np.array(np.sqrt(m_q1_tf.multiply(m_q1_tf).sum(axis=1)))[:, 0] * \
        np.array(np.sqrt(m_q2_tf.multiply(m_q2_tf).sum(axis=1)))[:, 0]
v_num[np.where(v_den == 0)] = 1
v_den[np.where(v_den == 0)] = 1

v_score = 1 - v_num/v_den

df['f_trigram_tfidf_cosine'] = v_score

# trigram_tfidf_l2_euclidean
tft = TfidfTransformer(
    norm='l2', 
    use_idf=True, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['f_trigram_tfidf_l2_euclidean'] = v_score

# trigram_tfidf_l1_euclidean 	
tft = TfidfTransformer(
    norm='l1', 
    use_idf=True, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)
v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['f_trigram_tfidf_l1_euclidean'] = v_score

# trigram_tf_l2_euclidean
tft = TfidfTransformer(
    norm='l2', 
    use_idf=False, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['f_trigram_tf_l2_euclidean'] = v_score


svd = TruncatedSVD(n_components=100)
m_svd = svd.fit_transform(sparse.csc_matrix(sparse.vstack((m_q1_tf, m_q2_tf))))    

with open('1_svd.pkl', 'wb') as f:
        pickle.dump(svd, f)
with open('1_m_svd.npz', 'wb') as f:
        np.savez(f, m_svd)

df['f_q1_q2_tf_svd0'] = m_svd[:, 0]
#df['f_q1_q2_tf_svd0'].to_csv('ms-m_q1_q2_tf_svd0.csv') 

ix_train = df[df['test_id'] == -1]
ix_test = df[df['test_id'] >= 0]


col = [c for c in ix_train.columns if c[:1]=='f']
ix_train.to_csv('train_ix.csv',index = False, columns = col)
ix_test.to_csv('test_ix.csv',index = False, columns = col)