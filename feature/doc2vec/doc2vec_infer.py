#Import Initial Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import gensim
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
import re 
from collections import namedtuple
import multiprocessing
import datetime
import os

tokenizer = RegexpTokenizer(r'\w+')
stopwords = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def Cosine(vec1, vec2):
    vec1 = np.array(vec1,dtype=np.float)
    vec2 = np.array(vec2,dtype=np.float)
    Lx = np.sqrt(vec1.dot(vec1))
    Ly = np.sqrt(vec2.dot(vec2))
    return vec1.dot(vec2) / ((Lx * Ly)+0.000001)

def Manhatton(vec1, vec2):
    return np.sum(np.fabs(np.array(vec1,dtype=np.float) - np.array(vec2,dtype=np.float)))

def Euclidean(vec1, vec2):
    return np.sqrt(np.sum(np.array(vec1,dtype=np.float) - np.array(vec2,dtype=np.float)) ** 2)

def PearsonSimilar(vec1, vec2):
    vec1 = np.array(vec1,dtype=np.float)
    vec2 = np.array(vec2,dtype=np.float)
    data = np.vstack((vec1, vec2))
    return pd.DataFrame(data).T.corr('pearson')[0][1]

def SpearmanSimilar(vec1, vec2):
    vec1 = np.array(vec1,dtype=np.float)
    vec2 = np.array(vec2,dtype=np.float)
    data = np.vstack((vec1, vec2))
    return pd.DataFrame(data).T.corr('spearman')[0][1]

def KendallSimilar(vec1, vec2):
    vec1 = np.array(vec1,dtype=np.float)
    vec2 = np.array(vec2,dtype=np.float)
    data = np.vstack((vec1, vec2))
    return pd.DataFrame(data).T.corr('kendall')[0][1]

def get_processed_text(text=""):
    """
    Remove stopword,lemmatizing the words and remove special character to get important content
    """
    clean_text = re.sub('[^a-zA-Z0-9 \n\.]', ' ', text)
    tokens = tokenizer.tokenize(clean_text)
    tokens = [lemmatizer.lemmatize(token.lower().strip()) for token in tokens
              if token not in stopwords and len(token) >= 2]
    return tokens

model_name = "%s" %("doc_3_vec_model3")
model_saved_file = "%s" %(model_name)
model = gensim.models.doc2vec.Doc2Vec.load(model_saved_file)

def get_question_vector(question1 = ""):
    question_token1 = get_processed_text(question1)
    tokenize_text1 = ' '.join(question_token1)
    tokenize_text1 = gensim.utils.simple_preprocess(tokenize_text1)
    infer_vector_of_question1 = model.infer_vector(tokenize_text1)
    return infer_vector_of_question1


test = pd.read_csv("./input/test.csv")[:]
train = pd.read_csv("./input/train.csv")[:]
train = train.fillna('empty')
test = test.fillna('empty')


def makeFeature(df_features):
    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S')
    print ('get sentence vector')
    df_features['doc2vec1'] = df_features.question1.map(lambda x: get_question_vector(x))
    df_features['doc2vec2'] = df_features.question2.map(lambda x: get_question_vector(x))
    print now.strftime('%Y-%m-%d %H:%M:%S')
    print ('get six kinds of coefficient about vector')
    df_features['z3_cosine'] = df_features.apply(lambda x: Cosine(x['doc2vec1'], x['doc2vec2']), axis=1)
    df_features['z3_manhatton'] = df_features.apply(lambda x: Manhatton(x['doc2vec1'], x['doc2vec2']), axis=1)
    df_features['z3_euclidean'] = df_features.apply(lambda x: Euclidean(x['doc2vec1'], x['doc2vec2']), axis=1)
    df_features['z3_pearson'] = df_features.apply(lambda x: PearsonSimilar(x['doc2vec1'], x['doc2vec2']), axis=1)
    df_features['z3_spearman'] = df_features.apply(lambda x: SpearmanSimilar(x['doc2vec1'], x['doc2vec2']), axis=1)
    df_features['z3_kendall'] = df_features.apply(lambda x: KendallSimilar(x['doc2vec1'], x['doc2vec2']), axis=1)
    now = datetime.datetime.now()
    print now.strftime('%Y-%m-%d %H:%M:%S')
    return df_features

if __name__ == "__main__":

    train = makeFeature(train)
    col = [c for c in train.columns if c[:1]=='z']
    train.to_csv('train_doc2vec3.csv', index=False, columns = col)
    test = makeFeature(test)
    test.to_csv('test_sdoc2vec3.csv', index=False, columns = col)
