'''
Single model may achieve LB scores at around 0.29+ ~ 0.30+
Average ensembles can easily get 0.28+ or less
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!

The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 2.7

According to experiments by kagglers, Theano backend with GPU may give bad LB scores while
        the val_loss seems to be fine, so try Tensorflow backend first please
'''

########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys, pickle

reload(sys)
sys.setdefaultencoding('utf-8')

########################################
## set directories and parameters
########################################
BASE_DIR = '../../input/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm, \
                                  rate_drop_dense)

MAX_SEQUENCE_LENGTH = 40
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

with open(BASE_DIR+'preprocess1.pkl') as f:
    (data_1, data_2, labels, test_data_1, test_data_2, test_ids, tokenizer) = pickle.load(f)

with open(BASE_DIR+'ab_features.pkl') as f:
    (abtrain, abtest) = pickle.load(f)

with open(BASE_DIR+'freq_features.pkl') as f:
    (freqtrain, freqtest) = pickle.load(f)
# freqtrain, freqtest = freqtrain.values, freqtest.values

# data_1 = data_1[:5,:]
# data_2 = data_2[:5,:]
# labels = labels[:5]
# abtrain = abtrain[:5,:]
# freqtrain = freqtrain[:5,:]
#
# test_data_1 = test_data_1[:5,:]
# test_data_2 = test_data_2[:5,:]
# abtest = abtest[:5,:]
# freqtest=freqtest[:5,:]

########################################
## index word vectors
########################################
print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
                                             binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))


########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## sample train/validation data
########################################
# np.random.seed(1234)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1) * (1 - VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1) * (1 - VALIDATION_SPLIT)):]

sequences_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
sequences_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
features_train = np.vstack((abtrain[idx_train,:], abtrain[idx_train,:]))
freq_train = np.vstack((freqtrain[idx_train, :], freqtrain[idx_train, :]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

sequences_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
sequences_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
features_val = np.vstack((abtrain[idx_val,:], abtrain[idx_val,:]))
freq_val = np.vstack((freqtrain[idx_val, :], freqtrain[idx_val, :]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val == 0] = 1.309028344

########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

features_input = Input(shape=(abtrain.shape[1],), dtype=np.float32)
freq_input = Input(shape=(freqtrain.shape[1],), dtype=np.float32)

merged = concatenate([x1, y1, features_input, freq_input])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

########################################
## add class weight
########################################
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input, features_input, freq_input], \
              outputs=preds)
model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
# model.summary()
print(STAMP)

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([sequences_1_train, sequences_2_train, features_train, freq_train], labels_train, \
                 validation_data=([sequences_1_val, sequences_2_val, features_val, freq_val], labels_val, weight_val), \
                 epochs=200, batch_size=2048, shuffle=True, \
                 class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

########################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

preds = model.predict([test_data_1, test_data_2, abtest, freqtest], batch_size=8192, verbose=1)
preds += model.predict([test_data_2, test_data_1, abtest, freqtest], batch_size=8192, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id': test_ids, 'is_duplicate': preds.ravel()})
submission.to_csv('%.4f_' % (bst_val_score) + STAMP + '.csv', index=False)
