#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
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

print "load start ..."
def sigmoid(x):
    return 1 / (1 + np.exp(0-x))


def discard_id(X):
    print np.shape(X)
    return X[:, 1:]


def predict(X_val, y_val):
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_val_origin = X_val
    X_val = discard_id(X_val)
    
    clf = joblib.load('sandwich_model.pkl')
    result = clf.predict_proba(X_val)
    print 'result:',result
    y_scores = []
    for line in result:
        y_scores.append(line[1])
    y_scores = np.array(y_scores)
    y_val = [int(i) for i in y_val]
    print "auc; ", roc_auc_score(y_val, y_scores)
    results = zip(X_val_origin, y_scores, y_val)
    results = sorted(results, key=lambda x: x[1], reverse=True)
    precision, recall, thresholds = precision_recall_curve(
        y_val, y_scores)

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall, precision, color='navy',
             label='Precision-Recall curve')
    plt.savefig("./result.pdf")
    return results

    
def boost(X_train, y_train, x_val, y_val):
    # Early-stopping
    
    X_train, y_train, X_val, y_val = np.array(X_train), np.array(y_train), np.array(x_val), np.array(y_val)
    #X_train, X_test, y_train, y_test = train_test_split(x_train, y_click_train, random_state=0, test_size=0.33)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=0, test_size=0.16)
    #pickle.dump((X_train, X_test, y_train, y_test, X_val, y_val), open("save.p", "wb"))
    
    #X_train, X_test, y_train, y_test, X_val, y_val = pickle.load(open("save.p", "rb"))
    X_val_origin = X_val
    X_val = discard_id(X_val)
    #print X_val[0:5]
    print "start_compute"
    
    num = 1
    result = []
    learning_rate_list = [0.05]
    subsample = [0.8]
    max_depth = [5]
    min_child_weight = [5]
    p = []
    for lr in learning_rate_list:
        for sb in subsample:
            for md in max_depth:
                for cd in min_child_weight:
                    p.append((lr, sb, md, cd))

    for i in range(len(p)):
        lr, sb, md, cd = p[i]
        print 'max_depth:', md, 'min_child_weight:', cd
        param_dist = {'n_estimators': 300, 'max_depth': md, 'subsample': sb, 'min_child_weight': cd, 'learning_rate': lr, 
                      'colsample_bytree': 0.9, 'colsample_bylevel':0.9}

        clf = xgb.XGBClassifier(**param_dist)
        clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="merror",
                eval_set=[(X_train, y_train), (X_test, y_test)])

        param_dist = {'n_estimators': clf.best_iteration + 1, 'max_depth': md, 'subsample': sb, 'min_child_weight': cd, 'learning_rate': lr,
                      'colsample_bytree': 0.9,
                      'colsample_bylevel': 0.9}
        clf = xgb.XGBClassifier(**param_dist)
        clf.fit(X_train, y_train, eval_metric="merror",
                eval_set=[(X_train, y_train), (X_test, y_test)])

        print 'importance', clf.feature_importances_

        joblib.dump(clf, 'sandwich_model.pkl')
        clf = joblib.load('sandwich_model.pkl')
        result1 = clf.predict_proba(X_val)
        #print 'result:',result1
        result1 = np.log(result1 / (1 - result1))
        if int(i) == 0:
            result = result1
        else:
            result += result1

    result = sigmoid((result) / num)
    y_scores = []
    for line in result:
        y_scores.append(line[1])
    y_scores = np.array(y_scores)
    y_val = [int(i) for i in y_val]
    #print "auc; ", roc_auc_score(y_val, y_scores)
    results = zip(X_val_origin, y_scores, y_val)
    results = sorted(results, key=lambda x: x[1], reverse=True)
    
    '''
    score_results = sorted(results, key=lambda x: x[0][0], reverse=True)
    during_results = sorted(results, key=lambda x: x[0][1], reverse=True)
                    
    truth_rate = float(len(filter(lambda x: x[2] == 1, results)))/len(results)
    tp_list = []
    score_tp_list = []
    during_tp_list = []
    x_list = []
    for i in range(2,101,2):
        x_list.append(i)
        choose_len = int(len(results) * 0.01 * i)
        # if (i == 5):
        #     for result in during_results[:20]:
        #         for j in result[0]:
        #             print '%f'%j
        #     print during_results[:20]
        #     print len(filter(lambda x: x[2] == 1, during_results[:20]))
        #     print choose_len
        #     print truth_rate
        
        tp = float(len(filter(lambda x: x[2] == 1, results[:choose_len])))/ choose_len
        tp_list.append(tp/truth_rate)
        score_tp = float(len(filter(lambda x: x[2] == 1, score_results[:choose_len])))/ choose_len
        score_tp_list.append(score_tp/truth_rate)
        during_tp = float(len(filter(lambda x: x[2] == 1, during_results[:choose_len])))/ choose_len
        during_tp_list.append(during_tp/truth_rate)
    '''
        
    '''    
    precision, recall, thresholds = precision_recall_curve(
        y_val, y_scores)

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall, precision, color='navy',
             label='Precision-Recall curve')
    plt.savefig("./result.pdf")
    '''
    return results
