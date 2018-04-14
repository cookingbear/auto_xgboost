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
from config_util import get_value

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.cross_validation import KFold, train_test_split
from sklearn.externals import joblib
import random
import pickle

print "load start ..."
def sigmoid(x):
    return 1 / (1 + np.exp(0-x))


def discard_id(X):
    print np.shape(X)
    return X[:, 1:]


def get_features(f):
    with open(f) as f:
        lines = f.readlines()
    return lines


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def plot_tree(clf):
    xgb.plot_tree(clf, num_trees=0, fmap='xgb.fmap')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(75, 50)
    fig.savefig("./tree.pdf")


def plot_importance(clf):
    xgb.plot_importance(clf)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(75, 50)
    fig.savefig("./importance.pdf")


def predict(X_test, y_test):
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test_origin = X_test
    X_test = discard_id(X_test)
    
    clf = joblib.load(get_value('model_name'))

    plot_importance(clf)
    plot_tree(clf)
    # prediction
    test_data = xgb.DMatrix(data=X_test, label=y_test, feature_names=get_feat_name())
    result = clf.predict(test_data)
    y_scores = []
    for line in result:
        y_scores.append(line)
    y_scores = np.array(y_scores)
    y_test = [int(i) for i in y_test]
    results = zip(X_test_origin, y_scores, y_test)
    return results


def get_feat_name():
    if not get_value('feats_name'):
        return None
    with open(get_value('feats_name')) as f:
        names = f.readlines()
    return names

    
def boost(data=None):
    print 'load_data..'
    if data:
        X_train, y_train, x_test, y_test = data
        X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(x_test), np.array(y_test)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.16)
        pickle.dump((X_train, X_test, y_train, y_test, X_val, y_val), open("save.p", "wb"))
    else:
        X_train, X_test, y_train, y_test, X_val, y_val = pickle.load(open("save.p", "rb"))
    
    X_test_origin = X_test
    X_test = discard_id(X_test)
    print "start_compute.."
    
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

    train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=get_feat_name())
    val_data = xgb.DMatrix(data=X_val, label=y_val, feature_names=get_feat_name())
    test_data = xgb.DMatrix(data=X_test, label=y_test, feature_names=get_feat_name())
    
    results = []
    for i in range(len(p)):
        lr, sb, md, cd = p[i]
        print 'max_depth:', md, 'min_child_weight:', cd
        param_dict = {'max_depth': md, 'subsample': sb, 'min_child_weight': cd, 'learning_rate': lr, 'num_class':int(get_value('num_class')),
                      'colsample_bytree': 0.9, 'colsample_bylevel':0.9, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss'}

        num_rounds = 300
        watchlist = [(train_data, 'train'), (val_data, 'val')]
        clf = xgb.train(param_dict, train_data, num_rounds, watchlist, early_stopping_rounds=20)
        # early_stopping
        print 'early_stopping:', clf.best_iteration + 1, clf.best_score
        clf = xgb.train(param_dict, train_data, clf.best_iteration + 1, watchlist, early_stopping_rounds=20)
        results.append((clf.best_score, clf))

    # choose best
    min = 1
    clf = None
    for r in results:
        if r[0] < min:
            min = r[0]
            clf = r[1]

    # importance
    ceate_feature_map(get_features(get_value('feats_name')))
    print 'importance', clf.get_score('xgb.fmap', importance_type=get_value('importance_type'))
    # save model
    joblib.dump(clf, get_value('model_name'))
    # prediction
    result = clf.predict(test_data)
    y_scores = []
    for line in result:
        y_scores.append(line)
    y_scores = np.array(y_scores)
    y_test = [int(i) for i in y_test]
    results = zip(X_test_origin, y_scores, y_test)
    return results
