import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.engine.topology import Merge
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text

data = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')
y = data.is_duplicate.values

tk = text.Tokenizer(nb_words=200000)


#设置每条句子的固定长度为40                                                                                     
max_len = 40
tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))
x1 = tk.texts_to_sequences(data.question1.values) 
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tk.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

word_index = tk.word_index #问句1和文句2得到的字典

ytrain_enc = np_utils.to_categorical(y)

embeddings_index = {} #读取词向量，构造dict
f = open('data/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#得到问句1和问句2构造的字典，所有词组成的词向量，shape 为（，300）
embedding_matrix = np.zeros((len(word_index) + 1, 300)) 
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector



max_features = 200000
filter_length = 5
nb_filter = 64
pool_length = 4

model = Sequential()
print('Build model...')

model1 = Sequential()
model1.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
#Embedding输出的是三维[batch_size, sequence, output_dim]

model1.add(TimeDistributed(Dense(300, activation='relu'))) #Dense表示全连接层，输出维度为300，激活函数为relu
model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,))) #输出的shape(*,300)
#将一个sequence按照行方向叠加，输出的维度为[batch_size, output_dim]，将每一个句子用一个维度为300维的向量表示

model2 = Sequential()
model2.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))

model2.add(TimeDistributed(Dense(300, activation='relu')))
model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

# model3 model4

model3 = Sequential()
model3.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
#输出shape为[batch_size, sequence, output_dim] (None, 40, 300)

model3.add(Convolution1D(nb_filter=nb_filter, #卷积核的数目，输出的维度 64
                         filter_length=filter_length, #卷积核长度，即取相邻词窗口的大小，5
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
#输出的shape为(None,36,64), 输出的维度为为卷积核的数目64，输出的序列长度为，40-5+1

model3.add(Dropout(0.2)) #在卷积输出单元，添加dropout 添加概率丢弃

model3.add(Convolution1D(nb_filter=nb_filter, #64
                         filter_length=filter_length, #5
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

#shape(None,32,64) # 36-5+1
model3.add(GlobalMaxPooling1D()) #最大池化
#shape[bath_size, output_dim] # shape(None,64)

model3.add(Dropout(0.2))
model3.add(Dense(300))
#经过全连接后维度又变为300, shape(None,300)

model3.add(Dropout(0.2))
model3.add(BatchNormalization())
#shape(None,300)


model4 = Sequential()
model4.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
model4.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model4.add(Dropout(0.2))

model4.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model4.add(GlobalMaxPooling1D())
model4.add(Dropout(0.2))

model4.add(Dense(300))
model4.add(Dropout(0.2))
model4.add(BatchNormalization())

#model5 model6
model5 = Sequential()
model5.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2)) #shape(Nnoe,40,300)
model5.add(LSTM(300, dropout_W=0.2, dropout_U=0.2)) #shape(None,300)

model6 = Sequential()
model6.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2))
model6.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

merged_model = Sequential()
merged_model.add(Merge([model1, model2, model3, model4, model5, model6], mode='concat'))
merged_model.add(BatchNormalization())
#输入的模型进行合并，横向连接，shape(None,1800)

merged_model.add(Dense(300)) #shape(None,300)
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(1))
merged_model.add(Activation('sigmoid')) #shape(None,1)

merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

merged_model.fit([x1, x2, x1, x2, x1, x2], y=y, batch_size=384, nb_epoch=200,
                 verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])

#这种方法属于并列结构，分别训练问句1和问句2的词向量，之后merge再合并一起训练