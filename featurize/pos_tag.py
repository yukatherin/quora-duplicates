from __future__ import division
import argparse, sys, time
import functools
from collections import defaultdict
import joblib
import scipy
import datetime
import operator
from sklearn.cross_validation import train_test_split
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


reload(sys)
sys.setdefaultencoding('utf-8')

def col_tagger(colmn_parsed):
    '''
    Args: 
        colmn_parsed: Pandas column
    Returns (column of lists of pos, column of space-delimited strs of pos)  
    '''
    t0 = time.time()
    NN = len(colmn_parsed)
    colmn_tokens = colmn_parsed.map(lambda x: [])
    colmn_tokens_str = colmn_parsed.map(lambda x: '')
    batch = 20000

    list_of_keys = colmn_parsed.keys()
    i = 0
    while i < NN:
        if i + batch < NN:
            end_row = i + batch
        else:
            end_row = NN
        str_batch = unicode(" ~ ".join(map(str, colmn_parsed[i:end_row])), errors='replace') # I checked that '~' is not in any of the questions in train/test
        tokens_batch = nltk.pos_tag(nltk.word_tokenize(str_batch))
        # str_batch=None

        k = i
        pos = 0
        while pos < len(tokens_batch):
            if tokens_batch[pos][0] == "~":
                colmn_tokens_str[list_of_keys[k]] = ' '.join(colmn_tokens[list_of_keys[k]])
                k += 1
            else:
                colmn_tokens[list_of_keys[k]].append(tokens_batch[pos][1])
            pos += 1
            # if (pos % 100000)==0:
            #    print ""+str(pos)+" out of "+str(len(tokens_all))+" tokens"
        colmn_tokens_str[list_of_keys[k]] = ' '.join(colmn_tokens[list_of_keys[k]])
        i += batch
        print "tagged " + str(min(i, NN)) + " out of " + str(NN) + " total rows; " + str(
            round((time.time() - t0) / 60, 1)) + " minutes"
    return colmn_tokens, colmn_tokens_str

if __name__ == "__main__":

    # load data
    BASEDIR = '../../input/'
    df_train = pd.read_csv(BASEDIR+'train.csv')
    df_train = df_train.fillna(' ')
    df_test = pd.read_csv(BASEDIR+'test.csv')

    # print a small sample
    for x in zip(col_tagger(df_train.question1[:4])[0].values, col_tagger(df_train.question2[:4])[0].values):
        print(x)
    for x in zip(col_tagger(df_train.question1[:4])[1].values, col_tagger(df_train.question2[:4])[1].values):
        print(x)

    # transform test pos
    test_pos_q1, test_posstr_q1 = col_tagger(df_test.question1)
    test_pos_q2, test_posstr_q2 = col_tagger(df_test.question2)
    with open(BASEDIR+'test_pos.bin', 'wb') as f:
        joblib.dump((test_pos_q1, test_pos_q2, test_posstr_q1, test_posstr_q2), f)
    # with open(BASEDIR+'test_pos.bin', 'rb') as f:
    #     (test_pos_q1, test_pos_q2, test_posstr_q1, test_posstr_q2) = joblib.load(f)

    # transform train pos
    train_pos_q1, train_posstr_q1 = col_tagger(df_train.question1)
    train_pos_q2, train_posstr_q2 = col_tagger(df_train.question2)
    with open(BASEDIR + 'train_pos.bin', 'wb') as f:
        joblib.dump((train_pos_q1, train_pos_q2, train_posstr_q1, train_posstr_q2), f)
    # with open(BASEDIR+'train_pos.bin', 'rb') as f:
    #     (train_pos_q1, train_pos_q2, train_posstr_q1, train_posstr_q2) = joblib.load(f)
    # with open(BASEDIR+'train_pos.bin', 'rb') as f:
    #     (train_pos_q1, train_pos_q2, train_posstr_q1, train_posstr_q2) = joblib.load(f)


    # fit count vectorizer
    print('fitting character model...')
    maxNumFeatures = 300000
    BagOfWordsExtractor = CountVectorizer(max_df=0.999, min_df=50, max_features=maxNumFeatures,
                                          analyzer='word', ngram_range=(1, 7), stop_words=None,
                                          binary=True, lowercase=True)

    print('done.')
    del train_pos_q1, train_pos_q2
    with open(BASEDIR+'BagOfWordsExtractor_pos_50_1_7_300k.bin', 'wb') as f:
        joblib.dump(BagOfWordsExtractor, f)
    with open(BASEDIR+'BagOfWordsExtractor_pos_50_1_7_300k.bin', 'rb') as f:
        BagOfWordsExtractor = joblib.load(f)

    trainQuestion_BOW_rep = BagOfWordsExtractor.fit_transform(pd.concat((train_posstr_q1, train_posstr_q2)))
    trainQuestion1_BOW_rep, trainQuestion2_BOW_rep = trainQuestion_BOW_rep[:df_train.shape[0], :], trainQuestion_BOW_rep[df_train.shape[0]:, :]
    testQuestion1_BOW_rep = BagOfWordsExtractor.transform(test_posstr_q1)
    testQuestion2_BOW_rep = BagOfWordsExtractor.transform(test_posstr_q2)
    assert trainQuestion1_BOW_rep.shape == trainQuestion2_BOW_rep.shape
    del test_pos_q1, test_pos_q2
    with open(BASEDIR+'BOW_rep_pos_50_1_7_300k.bin', 'wb') as f:
        joblib.dump((trainQuestion1_BOW_rep, trainQuestion2_BOW_rep, testQuestion1_BOW_rep, testQuestion2_BOW_rep), f)
    print('done')


