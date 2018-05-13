# encoding: utf-8

'''
@author: cookingbear

@contact: 664610407@qq.com

@file: thresholds.py

@time: 2018/5/12 下午5:49

'''

THRES = (0., 1)
STRIDE = 0.005

import numpy as np
from config_util import get_value


def __compute_matrix(results, threshold, num):
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


def __matrix_filter(matrix):
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
    best_matrix = __compute_matrix(results, threshold, num)
    while True:
        th = THRES[0]
        # threshold个位数循环计算
        while th < THRES[1]:
            threshold[i] = th
            th += STRIDE
            matrix = __compute_matrix(results, threshold, num)
            new_val = __matrix_filter(matrix)
            if best_score < new_val:
                best_score, best_threshold, best_matrix = (new_val, np.array(threshold), matrix)
        over = False
        # threshold十位数以上增加一次, 每增加一次后面的数清0, 首位数无法再增加则结束
        for i in range(num - 1):
            index = num - i - 2
            if threshold[index] + STRIDE < THRES[1]:
                threshold[index] += STRIDE
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