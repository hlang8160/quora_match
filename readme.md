## 用机器学习方法 
*`人工构造特征`*  

**Basic Feature Engineering**  

* Length of question1
* Length of question2
* Difference in the two lengths
* Character length of question1 without spaces
* Character length of question2 without spaces
* Number of words in question1
* Number of words in question2
* Number of common words in question1 and question2  

**The fuzzy features**    

* QRatio
* WRatio
* Partial ratio
* Partial token set ratio
* Partial token sort ratio
* Token set ratio
* Token sort ratio

**word2vec features**  


* Word mover distance
* Normalized word mover distance
* Cosine distance between vectors of question1 and question2
* Manhattan distance between vectors of question1 and question2
* Jaccard similarity between vectors of question1 and question2
* Canberra distance between vectors of question1 and question2
* Euclidean distance between vectors of question1 and question2
* Minkowski distance between vectors of question1 and question2
* Braycurtis distance between vectors of question1 and question2
* Skew of vector for question1
* Skew of vector for question2
* Kurtosis of vector for question1
* Kurtosis of vector for question2
---
***A separate set of w2v features consisted of vectors itself.**  

* Word2vec vector for question1
* Word2vec vector for question2  


**采用XGBoost模型，得到准取率为0.814**  


---
---
## 采用深度学习模型  


* 不人工构造特征，采用神经网络的方法分别构造问句1和问句2特征，得到6个模型，之后再联合进行训练（相当于match fuction）

* 利用已有的词向量glove.840B.300d.txt,构建深度神经网络架构
```
model1 = Sequential()
model1.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
```

* Embedding()参数

* input_dim：大或等于0的整数，字典长度，即输入数据最大下标+1，**输入的字典长度+1**
* output_dim：大于0的整数，代表全连接嵌入的维度， **词向量的维度**
* embeddings_initializer: 嵌入矩阵的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers 
* embeddings_regularizer: 嵌入矩阵的正则项，为Regularizer对象 
* embeddings_constraint: 嵌入矩阵的约束项，为Constraints对象 
* mask_zero：布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，该参数在使用递归层处理变长输入时有用。设置为True的话，模型中后续的层必须都支持masking，否则会抛出异常。如果该值为True，则下标0在字典中不可用，input_dim应设置为|vocabulary| + 2。 
* input_length：当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断。

* Embedding 层输入的shape为2D张量  形如（samples，sequence_length）的2D张量 

* Embedding 层输出shape为3D张量 形如(samples, sequence_length, output_dim)的3D张量
  

* Dense就是常用的全连接层，所实现的运算是output =activation(dot(input, kernel)+bias)。

* Dropout。Dropout将在训练过程中每次更新参数时随机断开一定百分比（rate）的输入神经元，Dropout层用于防止过拟合。
* Dropout用在神经网络中隐藏节点中，增加网络单元随机丢弃的几率。在每次batch中，每次训练得到的网络不同
* 增加网络的多样性，相当于bagging
* 与bagging 的不同是, 抽取的不同网络是共享参数，并不是完全独立，而bagging是针对不同的抽样数据集分别训练完全独立的网络。
  
  
* BatchNormalization层
* 该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1。
  
  [Is That a Duplicate Quora Question?](https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur)