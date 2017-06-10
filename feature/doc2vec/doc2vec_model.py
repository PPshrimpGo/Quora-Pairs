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

df_train = pd.read_csv("./input/train.csv")[:]
df_train_set1 = df_train[["qid1","question1"]]
df_train_set2 = df_train[["qid2","question2"]]
df_train_set1.columns = ["qid","question"]
df_train_set2.columns =["qid","question"]
df_train_set = pd.concat([df_train_set1,df_train_set2],axis=0)

#Language Processing
def get_processed_text(text=""):
    """
    Remove stopword,lemmatizing the words and remove special character to get important content
    """
    clean_text = re.sub('[^a-zA-Z0-9 \n\.]', ' ', text)
    tokens = tokenizer.tokenize(clean_text)
    tokens = [lemmatizer.lemmatize(token.lower().strip()) for token in tokens
              if token not in stopwords and len(token) >= 2]
    return tokens

#Process and clean up traing set
alldocuments = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')       
keywords = []
for index,record in df_train_set[:].iterrows():
    qid = str(record["qid"])
    question = str(record["question"])
    tokens = get_processed_text(question)
    words = tokens
    words_text = " ".join(words)
    words = gensim.utils.simple_preprocess(words_text)
    tags = [qid]
    alldocuments.append(analyzedDocument(words, tags))


def train_and_save_doc2vec_model(alldocuments,document_model="model4",m_iter=100,m_min_count=2,m_size=100,m_window=5):
    print ("Start Time : %s" %(str(datetime.datetime.now())))
    #Train Model
    cores = multiprocessing.cpu_count()
    abs_path = os.getcwd()
    saved_model_name = "doc_2_vec_%s" %(document_model)
    doc_vec_file = "%s" %(saved_model_name)
    if document_model == "model1":
        # PV-DBOW 
        model_1 = gensim.models.Doc2Vec(alldocuments,dm=0,workers=cores,size=m_size, window=m_window,min_count=m_min_count,iter=m_iter,dbow_words=1)
        model_1.save("%s" %(doc_vec_file))
        print ("model training completed : %s" %(doc_vec_file))
    elif document_model == "model2":
        # PV-DBOW 
        model_2 = gensim.models.Doc2Vec(alldocuments,dm=0,workers=cores,size=m_size, window=m_window,min_count=m_min_count,iter=m_iter,dbow_words=0)
        model_2.save("%s" %(doc_vec_file))
        print ("model training completed : %s" %(doc_vec_file))
    elif document_model == "model3":
        # PV-DM w/average
        model_3 = gensim.models.Doc2Vec(alldocuments,dm=1, dm_mean=1,size=m_size, window=m_window,min_count=m_min_count,iter=m_iter)
        model_3.save("%s" %(doc_vec_file))
        print ("model training completed : %s" %(doc_vec_file))

    elif document_model == "model4":
        # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
        model_4 = gensim.models.Doc2Vec(alldocuments,dm=1, dm_concat=1,workers=cores, size=m_size, window=m_window,min_count=m_min_count,iter=m_iter)
        model_4.save("%s" %(doc_vec_file))
        print ("model training completed : %s" %(doc_vec_file))
    print ("Record count %s" %len(alldocuments))
    print ("End Time %s" %(str(datetime.datetime.now())))

train_and_save_doc2vec_model(alldocuments)

def get_question_similarity_score(question1="",question2=""):
    print ("question1 - ",question1)
    print ("question2 - ",question2)
    model_name = "%s" %("doc_2_vec_model4")
    model_saved_file = "%s" %(model_name)
    model = gensim.models.doc2vec.Doc2Vec.load(model_saved_file)
    
    question_token1 = get_processed_text(question1)
    tokenize_text1 = ' '.join(question_token1)
    tokenize_text1 = gensim.utils.simple_preprocess(tokenize_text1)
    infer_vector_of_question1 = model.infer_vector(tokenize_text1)
    
    print("tokenize_text1",tokenize_text1,"infer_vector_of_question1",infer_vector_of_question1)
    
    question_token2 = get_processed_text(question2)
    tokenize_text2 = ' '.join(question_token2)
    tokenize_text2 = gensim.utils.simple_preprocess(tokenize_text2)
    infer_vector_of_question2 = model.infer_vector(tokenize_text2)
    
    print("tokenize_text2",tokenize_text2,"infer_vector_of_question2",infer_vector_of_question2)
    similarity_score = 1
    #similarity_score = model.docvecs.most_similar(infer_vector_of_question1)
    msg= "question : %s model_name : %s " %(question,model_name)
   
    return similarity_score


def get_question_vector(question1 = ""):
    model_name = "%s" %("doc_2_vec_model4")
    model_saved_file = "%s" %(model_name)
    model = gensim.models.doc2vec.Doc2Vec.load(model_saved_file)
    question_token1 = get_processed_text(question1)
    tokenize_text1 = ' '.join(question_token1)
    tokenize_text1 = gensim.utils.simple_preprocess(tokenize_text1)
    infer_vector_of_question1 = model.infer_vector(tokenize_text1)
