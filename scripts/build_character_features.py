from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

# adapted from https://www.kaggle.com/selfishgene/shallow-benchmark-0-31675-lb
import time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os, joblib
import scipy
from collections import defaultdict

from sklearn import model_selection
from sklearn import linear_model

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor

if __name__ == "__main__":

    # load data
    trainDF = pd.read_csv('input/train.csv')
    labels = np.array(trainDF.ix[:, 'is_duplicate'])
    testDF = pd.read_csv('input/test.csv')

    trainDF = trainDF.fillna('')
    testDF.ix[testDF['question1'].isnull(), ['question1', 'question2']] = ''
    testDF.ix[testDF['question2'].isnull(), ['question1', 'question2']] = ''
    test_ids = testDF['test_id'].values

    # setup count vectorizer
    maxNumFeatures = 300000
    # bag of letter sequences (chars)
    # BagOfWordsExtractor = CountVectorizer(max_df=0.999, min_df=50, max_features=maxNumFeatures,
    #                                       analyzer='char', ngram_range=(1, 10),
    #                                       binary=True, lowercase=True)
    BagOfWordsExtractor = CountVectorizer(max_df=0.999, min_df=30, max_features=maxNumFeatures,
                                          analyzer='char', ngram_range=(2, 10),
                                          binary=True, lowercase=False)
    # bag of words
    #BagOfWordsExtractor = CountVectorizer(max_df=0.999, min_df=10, max_features=maxNumFeatures,
    #                                      analyzer='word', ngram_range=(1,6), stop_words='english',
    #                                      binary=True, lowercase=True)

    # fit
    print('fitting character model...')
    BagOfWordsExtractor.fit(pd.concat((trainDF.ix[:,'question1'],trainDF.ix[:,'question2'])).unique())

    # save trained model
    with open('tmp/BagOfWordsExtractor_2_10_30_300k_uc.pkl', 'wb') as f:
        joblib.dump(BagOfWordsExtractor, f)
    '''
    with open('tmp/BagOfWordsExtractor_2_10_30_300k_uc.pkl', 'rb') as f:
        BagOfWordsExtractor = pickle.load(f)
    '''

    # transform
    print('transforming train...')
    trainQuestion1_BOW_rep = BagOfWordsExtractor.transform(trainDF.ix[:, 'question1'])
    print('done1')
    trainQuestion2_BOW_rep = BagOfWordsExtractor.transform(trainDF.ix[:, 'question2'])
    print('done2')
    print((trainQuestion2_BOW_rep.shape, trainQuestion1_BOW_rep.shape, labels.shape))

    with open('tmp/trainQuestion1_BOW_rep_trainQuestion2_BOW_rep_2_10_30_300k_uc.pkl', 'wb') as f:
        joblib.dump((trainQuestion1_BOW_rep, trainQuestion2_BOW_rep), f)

    print('transforming test...')
    testQuestion1_BOW_rep = BagOfWordsExtractor.transform(testDF.ix[:, 'question1'])
    print('done1')
    testQuestion2_BOW_rep = BagOfWordsExtractor.transform(testDF.ix[:, 'question2'])
    print('done2')

    with open('tmp/testQuestion1_BOW_rep_2_10_30_300k_uc.pkl', 'wb') as f:
         joblib.dump(testQuestion1_BOW_rep, f)
    with open('tmp/testQuestion2_BOW_rep_2_10_30_300k_uc.pkl', 'wb') as f:
         joblib.dump(testQuestion2_BOW_rep, f)


