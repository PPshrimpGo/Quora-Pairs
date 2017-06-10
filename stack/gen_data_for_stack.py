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

name = 'MLPC_'
split_len = 121662

result1 = pd.read_csv(name+'result_1.csv')
result2 = pd.read_csv(name+'result_2.csv')
result3 = pd.read_csv(name+'result_3.csv')
result4 = pd.read_csv(name+'result_4.csv')
result5 = pd.read_csv(name+'result_5.csv')

"""
切分的数据集都放在前面了。平均划分5份，最后一个多一条。没个文件的行数，包含index
2467459 LR_result_1.csv
2467459 LR_result_2.csv
2467459 LR_result_3.csv
2467459 LR_result_4.csv
2467460 LR_result_5.csv
"""
#拼接训练集
result = pd.concat([result1[0:split_len],result2[0:split_len],result3[0:split_len],result4[0:split_len],result5[0:split_len+1]])
result.to_csv(name+'train_result.csv',index=False)

avg1 = result1[split_len:]
avg2 = result2[split_len:]
avg3 = result3[split_len:]
avg4 = result4[split_len:]
avg5 = result5[split_len+1:]
print(avg1.shape[0])
print(avg2.shape[0])
print(avg3.shape[0])
print(avg4.shape[0])
print(avg5.shape[0])

def calc_avg(avg01,avg02,avg03,avg04,avg05):
	if np.isnan(avg01) or np.isnan(avg02) or np.isnan(avg03) or np.isnan(avg04) or np.isnan(avg05):
		print("wrong")
		return 0
	return (avg01+avg02+avg03+avg04+avg05)/5


df = pd.DataFrame()
df['fold1'] = avg1[name+'predictions']
df['fold2'] = avg2[name+'predictions']
df['fold3'] = avg3[name+'predictions']
df['fold4'] = avg4[name+'predictions']
df['fold5'] = avg5[name+'predictions']

print(df['fold1'].shape)
print(df['fold2'].shape)
print(df['fold3'].shape)
print(df['fold4'].shape)
print(df['fold5'].shape)
#平均5个测试集

df['test_id'] = np.arange(avg1.shape[0])
df[name+'is_duplicate'] = df.apply(lambda x:calc_avg(x['fold1'],x['fold2'],x['fold3'],x['fold4'],x['fold5']), axis=1)
print(df[name+'is_duplicate'].shape)
col = ['test_id', name+'is_duplicate']
df.to_csv(name+'test_result.csv',index=False, columns =col)
