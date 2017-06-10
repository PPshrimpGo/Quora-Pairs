import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict
import sys

train_orig =  pd.read_csv('./input/train.csv', header=0)
test_orig =  pd.read_csv('./input/test.csv', header=0)

kcores = pd.read_csv('./question_kcores.csv', header = 0)
ques = pd.concat([train_orig[['question1', 'question2']],test_orig[['question1', 'question2']]], axis=0).reset_index(drop='index')
q_dict = defaultdict(set)
for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

dict_questions = df_id.set_index('question').to_dict()
dict_questions = dict_questions["qid"]

new_id = 538000 # df_id["qid"].max() ==> 537933

def get_id(question):
    global dict_questions 
    global new_id 
    
    if question in dict_questions:
        return dict_questions[question]
    else:
        new_id += 1
        dict_questions[question] = new_id
        return new_id



train_orig['z_q1_q2_intersect'] = train_orig.apply(q1_q2_intersect, axis=1, raw=True)
test_orig['z_q1_q2_intersect'] = test_orig.apply(q1_q2_intersect, axis=1, raw=True)

train_orig['z_q1_q2_intersect'] = train_orig.apply(q1_q2_intersect, axis=1, raw=True)
test_orig['z_q1_q2_intersect'] = test_orig.apply(q1_q2_intersect, axis=1, raw=True)



col = [c for c in train_orig.columns if c[:1]=='z']
train_orig.to_csv('train_magic2.csv', index=False, columns = col)
test_orig.to_csv('test_magic2.csv', index=False, columns = col)
