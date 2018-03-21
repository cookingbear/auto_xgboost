#encoding:utf-8

import matplotlib
matplotlib.use('Agg')
import json
import matplotlib.pyplot as plt
from urllib import quote
import string
import time
import sys
import random

import cPickle
import xgboost as xgb

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.cross_validation import KFold, train_test_split
from sklearn.externals import joblib
import random
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier
import pickle


y_map = {'angry':0, 'sad':1, 'happy':2, 'fear':3, 'neutral':4, 'surprise':5}
y_map = {v: k for k, v in y_map.iteritems()}

def predict(X_val, keys):
    X_val = np.array(X_val)
    clf = joblib.load('sandwich_model.pkl')
    result = clf.predict_proba(X_val)
    y_scores = []
    
    with open('result.txt', 'w') as wf:
        for i, line in enumerate(result):
            wf.write(str(line[np.argmax(line)]))
            wf.write(' ')
            wf.write(str(y_map[np.argmax(line)]))
            wf.write(' ') 
            wf.write(keys[i])
            wf.write('\n')
            wf.flush()


with open('/home/q/opensmile/result.json') as f:
    x_val = []
    keys = []
    for line in f:
        x_val.append(json.loads(line)['result'])
        keys.append(json.loads(line)['key'].encode('utf-8'))
    predict(x_val, keys)

