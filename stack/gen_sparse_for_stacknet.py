#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import pandas as pd
import numpy as np
import datetime
import multiprocessing
import difflib
import sys
from scipy.sparse import csr_matrix,hstack

reload(sys)
sys.setdefaultencoding('utf-8')



def fromsparsetofile(filename, array, deli1=" ", deli2=":",ytarget=None):    
    zsparse=csr_matrix(array)
    indptr = zsparse.indptr
    indices = zsparse.indices
    data = zsparse.data
    print(" data lenth %d" % (len(data)))
    print(" indices lenth %d" % (len(indices)))    
    print(" indptr lenth %d" % (len(indptr)))
    
    f=open(filename,"w")
    counter_row=0
    for b in range(0,len(indptr)-1):
        #if there is a target, print it else , print nothing
        if ytarget!=None:
             f.write(str(ytarget[b]) + deli1)     
             
        for k in range(indptr[b],indptr[b+1]):
            if (k==indptr[b]):
                if np.isnan(data[k]):
                    f.write("%d%s%f" % (indices[k],deli2,-1))
                else :
                    if abs(data[k]-0.0000000)>0.000001 :
                        f.write("%d%s%f" % (indices[k],deli2,data[k]))                    
            else :
                if np.isnan(data[k]):
                    f.write("%s%d%s%f" % (deli1,indices[k],deli2,-1))  
                else :
                    if abs(data[k]-0.0000000)>0.000001 :
                        f.write("%s%d%s%f" % (deli1,indices[k],deli2,data[k]))
        f.write("\n")
        counter_row+=1
        if counter_row%10000==0:    
            print(" row : %d " % (counter_row))    
    f.close()  

origincol = ['id','qid1','qid2','is_duplicate','z_stem_tfidf','z_noun_match','z_match_ratio','z_word_match','z_tfidf_sum1','z_tfidf_sum2','z_tfidf_mean1','z_tfidf_mean2','z_tfidf_len1','z_tfidf_len2']#, 	z_word_match	z_tfidf_sum1	z_tfidf_sum2	z_tfidf_mean1	z_tfidf_mean2	z_tfidf_len1	z_tfidf_len2
origincol2= ['test_id','z_stem_tfidf','z_noun_match','z_match_ratio','z_word_match','z_tfidf_sum1','z_tfidf_sum2','z_tfidf_mean1','z_tfidf_mean2','z_tfidf_len1','z_tfidf_len2']#, 	z_word_match	z_tfidf_sum1	z_tfidf_sum2	z_tfidf_mean1	z_tfidf_mean2	z_tfidf_len1	z_tfidf_len2

copycol = ['z_dist','z_document_dis','z_tfidf_cos_sim','z_w2v_unique','z_w2v_dis_e','z_w2v_dis_mink','z_w2v_dis_cityblock','z_w2v_dis_canberra','z_q1_skew','z_q2_skew','z_q1_kur','z_q2_kur']
#copycol = ['z_dist','z_tfidf_cos_sim','z_w2v_dis_e']


##n-gram
copycol2 = ['f_1dis','f_2word_dis','f_2char_dis','f_3word_dis','f_3char_dis','f_1dis2','f_2word_dis2','f_2char_dis2','f_3word_dis2','f_3char_dis2','f_1dis3','f_2word_dis3','f_2char_dis3','f_3word_dis3','f_3char_dis3','f_1dis4','f_2word_dis4','f_2char_dis4','f_3word_dis4','f_3char_dis4','f_1dise','f_2word_dise','f_2char_dise','f_3word_dise','f_3char_dise','f_1dis2e','f_2word_dis2e','f_2char_dis2e','f_3word_dis2e','f_3char_dis2e','f_1dis3e','f_2word_dis3e','f_2char_dis3e','f_3word_dis3e','f_3char_dis3e','f_1dis4e','f_2word_dis4e','f_2char_dis4e','f_3word_dis4e','f_3char_dis4e',]

copycol3 = ['f_cosine','f_manhatton','f_euclidean','f_pearson','f_spearman','f_kendall','f_cosine_w2v','f_euclidean_w2v','f_manhatton_w2v','f_raw_jaccarc','f_raw_ochiai','f_raw_dice','f_expand_jaccarc','f_expand_ochiai','f_expand_dice','f_nouns_jaccarc','f_nouns_ochiai','f_nouns_dice','f_verbs_jaccarc','f_verbs_ochiai','f_verbs_dice','f_adjs_jaccarc','f_adjs_ochiai','f_adjs_dice']

#magic
copycol4 = ['q1_freq','q2_freq']

copycol5 = ['z_w2v_unique_dis_e_weight','z_w2v_unique_dis_e','z_w2v_unique_dis_mink_w','z_w2v_unique_dis_cityblock_w','z_w2v_unique_dis_canberra_w','z_w2v_unique_dis_mink','z_w2v_unique_dis_cityblock','z_w2v_unique_dis_canberra','z_q1_unique_skew_w','z_q2_unique_skew_w','z_q1_unique_kur_w','z_q2_unique_kur_w','z_q1_unique_skew','z_q2_unique_skew','z_q1_unique_kur','z_q2_unique_kur']

copycol6 =['z_wordnet']

copycol7 = ['z_q1_q2_intersect']

copycol8 = ['z_intersection_count']

copycol9 = ['z_q1_place_num','z_q2_place_num','z_q1_has_place','z_q2_has_place','z_place_match_num','z_place_mismatch_num','z_place_match','z_place_mismatch']

copycol10 = ['z1_cosine','z1_manhatton','z1_euclidean','z1_pearson','z1_spearman','z1_kendall']

copycol11 = ['z2_cosine','z2_manhatton','z2_euclidean','z2_pearson','z2_spearman','z2_kendall']

copycol12 = ['z3_cosine','z3_manhatton','z3_euclidean','z3_pearson','z3_spearman','z3_kendall']

copycol13 = ['f_total_unique_words','f_total_unq_words_stop','f_wc_diff','f_wc_ratio','f_wc_diff_unique','f_wc_ratio_unique','f_wc_diff_unique_stop','f_wc_ratio_unique_stop','f_same_start_word','f_char_diff','f_char_ratio','f_char_diff_unique_stop']

copycol14 = ['z_qid1_max_kcore','z_qid2_max_kcore']

copycol15 = ['q1_q2_wm_ratio']

copycol16 = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29']

trainfeaure_old = pd.read_csv('./train_feature.csv', usecols= origincol)[:]
testfeature_old = pd.read_csv('./test_feature.csv', usecols = origincol2)[:]
#old from test
trainfeaure_new = pd.read_csv('./trainweigtht.csv',usecols = copycol)[:]
testfeature_new = pd.read_csv('./testweight.csv',usecols = copycol)[:]
# trainfeaure_new = pd.read_csv('./train_weight_tfidf.csv',usecols = copycol)[:]
# testfeature_new = pd.read_csv('./test_weight_tfidf.csv',usecols = copycol)[:]

trainfeaure_new2 = pd.read_csv('./train_gram_feature.csv',usecols = copycol2)[:]
testfeature_new2 = pd.read_csv('./test_gram_feature.csv',usecols = copycol2)[:]

#trainfeaure_new3 = pd.read_csv('./train_bagofwords350.csv')[:]
#testfeature_new3 = pd.read_csv('./test_bagofwords350.csv')[:]

#trainfeaure_new4 = pd.read_csv('./train_string_based_feature.csv',usecols =copycol3)[:]
#testfeature_new4 = pd.read_csv('./test_string_based_feature.csv',usecols =copycol3)[:]

trainfeaure_new5 = pd.read_csv('./train_freq.csv',usecols =copycol4)[:]
testfeature_new5 = pd.read_csv('./test_freq.csv',usecols =copycol4)[:]

trainfeaure_new6 = pd.read_csv('./train_weight_noweight.csv',usecols =copycol5)[:]
testfeature_new6 = pd.read_csv('./test_weight_noweight.csv',usecols =copycol5)[:]

trainfeaure_new7 = pd.read_csv('./train_wordnet.csv',usecols =copycol6)[:]
testfeature_new7 = pd.read_csv('./test_wordnet.csv',usecols =copycol6)[:]

trainfeaure_new8 = pd.read_csv('./train_magic2.csv',usecols =copycol7)[:]
testfeature_new8 = pd.read_csv('./test_magic2.csv',usecols =copycol7)[:]

#trainfeaure_new9 = pd.read_csv('./train_magic3.csv',usecols =copycol8)[:]
#testfeature_new9 = pd.read_csv('./test_magic3.csv',usecols =copycol8)[:]


#trainfeaure_new10 = pd.read_csv('./place_train_matches.csv',usecols =copycol9)[:]
#testfeature_new10 = pd.read_csv('./place_test_matches.csv',usecols =copycol9)[:]

#trainfeaure_new11 = pd.read_csv('./train_doc2vec.csv',usecols =copycol10)[:]
#testfeature_new11 = pd.read_csv('./test_sdoc2vec.csv',usecols =copycol10)[:]

#trainfeaure_new12 = pd.read_csv('./train_doc2vec2.csv',usecols =copycol11)[:]
#testfeature_new12 = pd.read_csv('./test_sdoc2vec2.csv',usecols =copycol11)[:]


#trainfeaure_new13 = pd.read_csv('./train_doc2vec3.csv',usecols =copycol12)[:]
#testfeature_new13 = pd.read_csv('./test_sdoc2vec3.csv',usecols =copycol12)[:]

trainfeaure_new14 = pd.read_csv('./train_simple.csv',usecols =copycol13)[:]
testfeature_new14 = pd.read_csv('./test_simple.csv',usecols =copycol13)[:]

trainfeaure_new15 = pd.read_csv('./train_kcore.csv',usecols =copycol14)[:]
testfeature_new15 = pd.read_csv('./test_kcore.csv',usecols =copycol14)[:]

trainfeaure_new16 = pd.read_csv('./new_magic_train.csv',usecols =copycol15)[:]
testfeature_new16 = pd.read_csv('./new_magic_test.csv',usecols =copycol15)[:]

trainfeaure_new17 = pd.read_csv('./train_ix.csv')[:]
testfeature_new17 = pd.read_csv('./test_ix.csv')[:]

trainfeaure_new18 = pd.read_csv('./train_svd.csv',usecols =copycol16)[:]
testfeature_new18 = pd.read_csv('./test_svd.csv',usecols =copycol16)[:]


train = trainfeaure_old 
test = testfeature_old
#z_dist	z_tfidf_cos_sim	z_w2v_nones	z_w2v_dis_e	z_w2v_dis_mink	z_w2v_dis_cityblock	z_w2v_dis_canberra	z_q1_skew	z_q2_skew	z_q1_kur	z_q2_kur
print "read 1 file"
for key in copycol:
	train[key] = trainfeaure_new[key]
	test[key] = testfeature_new[key]

print "read 2 file"
for key in copycol2:
	train[key] = trainfeaure_new2[key]
	test[key] = testfeature_new2[key]

#print "read 3 file"
#for key in [c for c in trainfeaure_new3.columns]:
#	train[key] = trainfeaure_new3[key]
#	test[key] = testfeature_new3[key]

#print "read 4 file"
#for key in copycol3:
# 	train[key] = trainfeaure_new4[key]
# 	test[key] = testfeature_new4[key]

print "read 5 file"
for key in copycol4:
	train['f_'+key] = trainfeaure_new5[key]
	test['f_'+key] = testfeature_new5[key]

print "read 5 file"
for key in copycol5:
	train[key] = trainfeaure_new6[key]
	test[key] = testfeature_new6[key]


print "read 7 file"
for key in copycol7:
	train[key] = trainfeaure_new8[key]
	test[key] = testfeature_new8[key]


# print "read 8 file"
# for key in copycol8:
# 	train[key] = trainfeaure_new9[key]
# 	test[key] = testfeature_new9[key]



# print "read 8 file"
# for key in copycol9:
# 	train[key] = trainfeaure_new10[key]
# 	test[key] = testfeature_new10[key]


#print "read 9 file"
#for key in copycol10:
#	train[key] = trainfeaure_new11[key]
#	test[key] = testfeature_new11[key]


#print "read 10 file"
#for key in copycol11:
#	train[key] = trainfeaure_new12[key]
#	test[key] = testfeature_new12[key]

#print "read 10 file"
#for key in copycol12:
#	train[key] = trainfeaure_new13[key]
#	test[key] = testfeature_new13[key]


for key in copycol13:
	train[key] = trainfeaure_new14[key]
	test[key] = testfeature_new14[key]

for key in copycol14:
	train[key] = trainfeaure_new15[key]
	test[key] = testfeature_new15[key]


for key in copycol15:
	train['f_'+key] = trainfeaure_new16[key]
	test['f_'+key] = testfeature_new16[key]

print "read 17 file"
for key in [c for c in trainfeaure_new17.columns]:
	train[key] = trainfeaure_new17[key]
	test[key] = testfeature_new17[key]

for key in copycol16:
	train['f_'+key] = trainfeaure_new18[key]
	test['f_'+key] = testfeature_new18[key]
# print "read 5 file"
# for key in copycol6:
# 	train[key] = trainfeaure_new7[key]
# 	test[key] = testfeature_new7[key]	
# train['z_dist'] = trainfeaure_new['z_dist']
# train['z_tfidf_cos_sim'] = trainfeaure_new['z_tfidf_cos_sim']
# train['z_w2v_nones'] = trainfeaure_new['z_w2v_nones']
# train['z_w2v_dis_e'] = trainfeaure_new['z_w2v_dis_e']
# train['z_w2v_dis_mink'] = trainfeaure_new['z_w2v_dis_mink']


# test['z_dist'] = testfeature_new['z_dist']
# test['z_tfidf_cos_sim'] = testfeature_new['z_tfidf_cos_sim']
# test['z_w2v_nones'] = testfeature_new['z_w2v_nones']
# test['z_w2v_dis_e'] = testfeature_new['z_w2v_dis_e']
# test['z_w2v_dis_e'] = testfeature_new['z_w2v_dis_e']


col = [c for c in train.columns if (c[:1] == 'z' or c[:1] == 'f')]

pos_train = train[train['is_duplicate'] == 1]
neg_train = train[train['is_duplicate'] == 0]

scale = 0.8
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
train = pd.concat([pos_train, neg_train])

X = csr_matrix(train[col])
X_test = csr_matrix(test[col])
y = train['is_duplicate'].values
print (X.shape, X_test.shape, y.shape) 
    
#export sparse data to stacknet format (which is Libsvm format)
fromsparsetofile("train.sparse", X, deli1=" ", deli2=":",ytarget=y)    
fromsparsetofile("test.sparse", X_test, deli1=" ", deli2=":",ytarget=None) 
print "done!"
