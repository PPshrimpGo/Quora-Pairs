## Quora: Questions pairs 总结
### 一、数据清洗总结

- fillna('EMPTY')对异常值得处理
- 停用词，标点符号，专有符号的去除和转换
- 词干化，词元化

实验的结果发现，预处理对于不同的特征有好有坏，所以对不同的特征选择了不同处理的方式。



### 二、特征总结
#### 字符和gram层次的特征
- bagofwords

使用了sklearn.feature_extraction.text中的CountVectorizer，生成了Bagofwords特征，实验之后选的参数是ngram(1,3)。
具体的代码在bagofwords.py里面。使用一个向量标注了两问题对应bagofwords的不同，作为一个特征输入到模型里，特征的维度是400.

最后模型使用了改特征。

- n-gram@word and char level

使用了1-3的字符和单词的gram，计算了各种距离，其中包括simhash<http://yanyiwu.com/work/2014/01/30/simhash-shi-xian-xiang-jie.html>
,还有Ochiai，Dice，Jaccarc在集合层次计算相似度的一些衡量手段。

- 基本的句子（字符和单词）统计特征

分词性统计了名词，动词相同的个数，基本的句子长度，词的数量，相同词的匹配率，tfidf的求和，平均，长度（后来思考一下这些特征做一些多项式组合可能会更好。）针对有无停用词的相同比率，相同词的数量不同词的数量等。还有针对字符级别的异同数量统计

#### 词向量表示句子的特征

- 词向量对句子进行相似词的扩充

拿到扩充的句子，进行了一次有Ochiai，Dice，Jaccarc等集合层次上的相似度计算。

- 求句子的表示，计算向量的距离和分布特征

句子的表示采用了两种种方法，一种是Bagofwords平均，一种是是Bagofwords的tfidf加权平均。拿到句子表示之后，计算了向量的距离：cosine,manhatton,euclidean，pearson，spearman，kendall

- 论文From wordembedding to docunment distacne的距离计算方法

- doc2vec产生词向量

PV-DBOW，PV-DM w/average，PV-DM w/concatenation - window=5 得到向量之后衡量相似度，这部分特征最终没有使用。、

- （1-3gram） tfidf的向量
拿到向量后计算距离

#### magic feature
 - 1、问题的频度
 - 2、基于问题构建的图，计算共同临接点的数量
 - 3、在2的基础上做word_match_share的加权
这两个提升很大，个人感觉第一个magic仔细研究数据可以发现，这也是说明了对数据做explore的重要性，第二个发现起来比较困难。

#### 其他的一些特征
- 1、kcore <https://www.kaggle.com/c/quora-question-pairs/discussion/33371>
- 2、pagerank <https://www.kaggle.com/shubh24/pagerank-on-quora-a-basic-implementation>
- 3、wordnet 计算相似度
> Wordnet is a huge library of synsets for almost all words in the English dictionary. The synsets for each word describe its meaning, part of speeches, and synonyms/antonyms. The synonyms help in identifying the semantic meaning of the sentence, when all words are taken together.
- 对tfidf矩阵做svd矩阵分解

### 三、模型总结
以上所有特征输入到XGBoost中，其中针对负类的做了百分之80的过采样，参数如下：
> {'colsample_bytree': 0.7, 'silent': 1, 'eval_metric': 'logloss', 'min_child_weight': 1, 'subsample': 0.8, 'eta': 0.05, 'gama': 0.005, 'objective': 'binary:logistic', 'seed': 1632, 'max_depth': 8}
单个模型取得最好的成绩是0.14090

### 四、stack和ensembel

第一层模型我们采用了xgboost(0.14),LoesticRegression(0.19),RandomForestClassifier(0.19), GradientBoostingClassifier(0.19),LSTM(0.20)这五个模型（括号内为LeaderBoard的score）。做了5折stack，效果不理想，个人思考不同的模型使用了相同的特征，模型之间的diversity很小。最后提交使用了LSTM和XGB的平均，这个LSTM后来发现其效果是0.2，blending的结果不是很差可能是因为模型的diversity。
最终的结果是 0.5XGB+0.5LSTM 提升到0.14033

### 五、经验教训

- 特征管理

只调优一个单个模型，没有考虑特征之间的相互作用，以及一些多项式组合，舍弃了一些特征，只保留了一个模型。

- stack

进行的太晚，发现问题（相关性太强）的时候已经来不及了，总的来说，stack应该取得不错的提升。

- 调参

进行的太晚， 应该同步使用多个模型，多种特征进行实验，一头扎在xgb上了。