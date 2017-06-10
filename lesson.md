## Quora: Questions pairs 总结

### 一、数据清洗总结

- fillna('EMPTY')对异常值得处理
- 停用词，标点符号，专有符号的去除和转换
- 词干化，词元化

在实验过程中，会经常性地出现一些乱码。后来发现是一些印度语符号，于是进行了特殊处理。

实验的结果发现，预处理对于不同的特征有好有坏，所以对不同的特征选择了不同处理的方式。

### 二、特征总结
#### 字符和gram层次的特征
- bagofwords

使用了sklearn.feature_extraction.text中的CountVectorizer，生成了Bagofwords特征，实验之后选的参数是ngram(1,3)。
具体的代码在bagofwords.py里面。使用一个向量标注了两问题对应bagofwords的不同，作为一个特征输入到模型里，选取的特征的维度是400。可以考虑对这个特征特征做矩阵分解的降维处理，占了整个树模型的大部分参数。

最后模型使用了该特征。

- n-gram @ word and char level

使用了1-3的字符和单词的gram，计算了各种距离，其中包括[simhash](http://yanyiwu.com/work/2014/01/30/simhash-shi-xian-xiang-jie.html)
,还有Ochiai，Dice，Jaccarc在集合层次计算相似度的一些衡量手段。

- 基本的句子（字符和单词）统计特征

分词性统计了名词，动词相同的个数，基本的句子长度，词的数量，相同词的匹配率，tfidf的求和，平均，长度（后来思考一下这些特征做一些多项式组合可能会更好。）针对有无停用词的相同比率，相同词的数量不同词的数量等。还有针对字符级别的异同数量统计。

#### 词向量表示句子的特征

- 词向量对句子进行相似词的扩充

相似词扩充使用nltk库中的brown语料。拿到扩充的词bag，进行了一次有Ochiai，Dice，Jaccarc等集合层次上的相似度计算。

- 求句子的表示，计算向量的距离和分布特征

句子的表示采用了两种种方法，一种是Bagofwords的直接平均，一种是是Bagofwords的tfidf加权平均。拿到句子表示之后，计算了向量的距离：cosine,manhatton,euclidean。以及数据特征：pearson，spearman，kendall


- doc2vec产生词向量

PV-DBOW，PV-DM w/average，PV-DM w/concatenation - window=5 得到向量之后衡量相似度，这部分特征最终没有使用。

- （1-3gram） tfidf的向量

拿到向量后计算距离

- 直接计算两个text之间的距离

1 编辑距离
2 论文From wordembedding to docunment distacne的距离计算对text中的每个词，找到另一个text中最近的词，求出距离，最后再做平均。将两个句子进行双向处理，获得两个距离，再取平均。


#### magic feature
 - 1、问题的频度
 - 2、基于问题构建的图，计算相同临接点的数量
 - 3、在2的基础上做word_match_share的加权
这两个提升很大，个人感觉第一个magic仔细研究数据可以发现，这也是说明了对数据做explore的重要性，第二个发现起来比较困难。

#### 其他的一些特征
- 1 [kcore](https://www.kaggle.com/c/quora-question-pairs/discussion/33371)
- 2 [pagerank](https://www.kaggle.com/shubh24/pagerank-on-quora-a-basic-implementation)
- 3 wordnet 计算相似度
> Wordnet is a huge library of synsets for almost all words in the English dictionary. The synsets for each word describe its meaning, part of speeches, and synonyms/antonyms. The synonyms help in identifying the semantic meaning of the sentence, when all words are taken together.
- 4 对tfidf矩阵做svd矩阵分解

### 三、模型总结

#### XGBoost
以上所有特征输入到XGBoost中，参数如下：
```
{
    'colsample_bytree': 0.7,
    'silent': 1,
    'eval_metric': 'logloss',
    'min_child_weight': 1,
    'subsample': 0.8,
    'eta': 0.05,
    'gama': 0.005,
    'objective': 'binary:logistic',
    'seed': 1632,
    'max_depth': 8
}
```
单个模型取得最好的成绩是0.14090
#### LSTM

LSTM的输入是sequence和去除了bagofwords和magic特征的组合。在CV上表现大约为0.12，但是最后还是过拟合挺严重，PB上大概0.20。

#### 过采样和rescale
其中针对重复的问题的做了百分之80的过采样，对结果提升较大。结果没有进行rescale，没有这方面的经验，经过尝试效果一般。


### 四、stack和ensemble

#### Stacking

> Base Model
```
{
    xgboost(0.14),
    LoesticRegression(0.19),
    RandomForestClassifier(0.19),
    GradientBoostingClassifier(0.19),
    LSTM(0.20)
}
```

> Level 2 Model
```
{
    xgboost(0.17)
}
```

做了5折stacking，效果不理想，表现还不如单个xgboost。后来把5个base model的结果作为特征加入单个xgboost，效果依然不如原始模型。

个人思考不同的模型使用了相同的特征，模型之间的diversity很小。
最后提交使用了LSTM（不包含magic feature）和XGB的平均，这个LSTM后来发现其效果是0.2，blending的结果不是很差可能是因为模型的diversity。
最终的结果是 0.5XGB+0.5LSTM 进行ensemble，Public LB 提升到0.14033。

### 五、经验教训

- 特征管理

1 没有对特征进行分类管理
2 只调优一个单个模型，没有考虑特征之间的相互作用
3 没有考虑特征之间的多项式组合，舍弃了一些特征，只保留了一个模型。
4 没有对一些特征结合数据分布做深入的思考，尤其是图方面的特征。顺着这个思路下去可以挖掘比较深。第一次比赛，过分与侧重问题本身，即“相似性”。对于数据的特征探索有所忽略。

- stacking

进行的太晚，发现问题（相关性太强）的时候已经来不及了，总的来说，stack应该取得不错的提升。

- 调参

进行的太晚， 应该同步使用多个模型，多种特征进行实验，一头扎在xgb上了。

- Pipeline

应该在比赛初期建立Pipeline。后期再产生新想法时，可以很快地执行
