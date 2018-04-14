# -*- coding: utf-8 -*-
import csv
import json
from config_util import get_value

import numpy as np
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
from boost import boost, predict
import cPickle


# 准备数据
def add_label():
    X_train, Y_train, x_test, y_test = create_feats()
    print "X_train_num, X_test_num:",len(X_train),len(x_test)
    return X_train, Y_train, x_test, y_test


if __name__ == "__main__":

    if get_value('mode') == 'train':
        if get_value('data_prepared') != 'true':
            x_train, y_train, x_test, y_test = add_label()
            results = boost((x_train, y_train, x_test, y_test))
        else:
            results = boost()
    else:
        _, _, x_test, y_test = add_label()
        results = predict(x_test, y_test)

    with open("result.txt", 'w') as wf:
        for line in results:
            for ele in line:
                wf.write(str(ele))
                wf.write(' ')
            wf.write('\n')
            wf.flush()
    
    classes = int(get_value('num_class'))
    matrix = [[0] * classes for i in range(classes)]
    preds = [0] * classes
    trues = [0] * classes
    with open('matrix/matrix.txt', 'w') as wf:
        for result in results:
            total = result[1][0]*1.5 + result[1][1] + result[1][2]*0.4
            pred = np.argmax([result[1][0]*1.5 / total, (result[1][1])/total, (result[1][2]*0.4)/total])
            #total = result[1][0]*0.95 + result[1][1]*1.4 + result[1][2]
            #pred = np.argmax([result[1][0]*0.95 / total, (result[1][1]*1.4)/total, (result[1][2])/total])
            #pred = np.argmax(np.array(result[1]))
            true = result[2]
            matrix[pred][true] += 1
            if true != 0 and pred == 0:
                print ([result[1][0]*1.5 / total, (result[1][1])/total, (result[1][2]*0.4)/total]), pred, true, 'http://100.84.164.142:1234/emotion_test_data/' + result[0][0].split('/')[-1]
            preds[pred] += 1
            trues[true] += 1
            wf.write('http://100.84.164.142:1234/emotion_test_data/' + result[0][0].split('/')[-1]+ ' ')
            for r in result[1]:
                wf.write(str(r)+" ")
            wf.write('\n')
            wf.flush()
    print matrix
    print preds
    print trues
