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

thres = (0., 1)
stride = 0.005

# 准备数据
def add_label():
    X_train, Y_train, x_test, y_test = create_feats()
    print "X_train_num, X_test_num:",len(X_train),len(x_test)
    return X_train, Y_train, x_test, y_test


def compute_matrix(results, threshold, num):
    matrix = np.array([[0] * num for i in range(num)])
    for result in results:
        true = result[2]  
        preds = np.array(result[1])
        preds -= threshold
        preds = np.where(preds > 0)[0]
        if len(preds) == 0:
            pred = 2
        else:
            pred = preds[0]
        matrix[pred][true] += 1
    return matrix


def matrix_filter(matrix):
    for i in range(len(matrix)):
        if np.sum(matrix[i]) == 0:
            return -1
    if float(matrix[0][0]) / np.sum(matrix[0]) < 0.2:
        return -1
    if float(matrix[1][1]) / np.sum(matrix[1]) < 0.5:
        return -1
    if float(matrix[1][1]) / np.sum(matrix[:, 1]) < 0.5:
        return -1
    return float(matrix[0][0]) / np.sum(matrix[:, 0])


def threshold_adjust(results):
    num = int(get_value('num_class'))
    threshold = [0 for i in range(num)]
    best_threshold = np.array(threshold)
    best_score = -1
    best_matrix = compute_matrix(results, threshold, num)
    while True:
        th = thres[0]
        # threshold个位数循环计算
        while th < thres[1]:
            threshold[i] = th
            th += stride
            matrix = compute_matrix(results, threshold, num)
            new_val = matrix_filter(matrix)
            if best_score < new_val:
                best_score, best_threshold, best_matrix = (new_val, np.array(threshold), matrix)  
        over = False
        # threshold十位数以上增加一次, 每增加一次后面的数清0, 首位数无法再增加则结束
        for i in range(num - 1):
            index = num - i - 2
            if threshold[index] + stride < thres[1]:
                threshold[index] += stride
                for j in range(num - index - 1):
                    # 后面的清0
                    threshold[index + j + 1] = 0
                break
            elif index == 0:
                over = True
                break
        if over:
            break

    print 'best_matrix:\n', best_matrix
    print 'best_threshold:', best_threshold
    print 'best_score:', best_score
    return best_score, best_threshold, best_matrix
 

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
    
    threshold_adjust(results)
    with open('matrix/matrix.txt', 'w') as wf:
        for result in results:
            wf.write('http://100.84.164.142:1234/emotion_test_data/' + result[0][0].split('/')[-1]+ ' ')
            for r in result[1]:
                wf.write(str(r)+" ")
            wf.write('\n')
            wf.flush()
