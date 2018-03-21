# -*- coding: utf-8 -*-
import csv
import json

import xlrd
import sys

sys.path.append('./')

import os
import jieba
import jieba.analyse
import jieba.posseg as pseg
import sys
import string
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from emotion_feat import create_feats
# import snownlp
from boost import boost, predict
import cPickle


# 准备数据
def add_label():
    X_train, Y_train, x_val, y_val = create_feats()
    #数据特征存储
    with open('feats.pkl', 'w') as f:
        cPickle.dump((X_train, Y_train, x_val, y_val), f)
    with open('feats.pkl', 'r') as f:
        X_train, Y_train, x_val, y_val = cPickle.load(f)

    print "X_train,Y_train:",len(X_train),len(Y_train)#,X_train[0:10],Y_train[0:10]
    return X_train, Y_train, x_val, y_val


if __name__ == "__main__":

    X_train, Y_train, x_val, y_val = add_label()
    results = boost(X_train, Y_train, x_val, y_val)
    #results = predict(x_val, y_val)
    with open("result.txt", 'w') as wf:
        for line in results:
            for ele in line:
                wf.write(str(ele))
                wf.write(' ')
            wf.write('\n')
            wf.flush()
