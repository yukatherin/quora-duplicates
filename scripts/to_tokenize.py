
from __future__ import unicode_literals
import pandas as pd
import multiprocessing as mp
# from nltk.tokenize.stanford import StanfordTokenizer
import sys, re, json, csv, time, pickle
sys.path.append('~/quora-duplicates/')
import enchant

reload(sys)
sys.setdefaultencoding('utf-8')

BASEDIR = '/Users/katherineyu/quora/'

print 'loading data...'
quora = pd.read_csv(BASEDIR + 'input/train.csv', sep=',', encoding='utf-8')
print quora.shape

quora.question1.to_csv(BASEDIR + 'data/train_q1.csv', header=None, index=False)