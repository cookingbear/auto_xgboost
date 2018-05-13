# encoding: utf-8

'''
@author: cookingbear

@contact: 664610407@qq.com

@file: main.py

@time: 2018/5/12 下午5:41

'''

import sys

import numpy as np

from config_util import get_value

sys.path.append('./')

from emotion_feat import create_feats
from boost import boost, predict
from thresholds import threshold_adjust


# 准备数据
def __add_label():
    X_train, Y_train, x_test, y_test = create_feats()
    print "X_train_num, X_test_num:", len(X_train), len(x_test)
    return X_train, Y_train, x_test, y_test


if __name__ == "__main__":
    if get_value('mode') == 'train':
        if get_value('data_prepared') != 'true':
            x_train, y_train, x_test, y_test = __add_label()
            results = boost((x_train, y_train, x_test, y_test))
        else:
            results = boost()
    else:
        _, _, x_test, y_test = __add_label()
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
            wf.write('http://100.84.164.142:1234/emotion_test_data/' + result[0][0].split('/')[-1] + ' ')
            for r in result[1]:
                wf.write(str(r) + " ")
            wf.write('\n')
            wf.flush()
