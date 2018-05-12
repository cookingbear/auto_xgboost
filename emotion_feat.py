#encoding:utf-8
import csv


# feature: during, R 0 or L 1,  callTimes, leftwordcount, rightwordcount, leftduration, rightdurtion,
# kf_positive_word, kf_negative_word, user_positive_word, user_negative_word
# positive_pos, positive_score, negative_pos, negative_score, left_size, right_size

import json

import math
from collections import defaultdict
import jieba
from gensim import corpora, models, similarities
import traceback
import sys

import soundfile
import numpy as np
from numpy.lib.stride_tricks import as_strided
from pydub import AudioSegment

from config_util import get_value


y_map = {'angry':0, 'sad':1, 'happy':2, 'fear':3, 'neutral':4, 'surprise':5}


def create_feats():
    X_train, Y_train, x_test, y_test = [], [], [], []
    train_part = False
    if get_value('all_test') == 'True':
        test_line = 1000000
    else:
        test_line = int(get_value('test_num'))
    feat_map = {}
    with open("/home/q/ruixiong.zhang/emotion_file/human_data/20180419_test_train.json", 'r') as f, open('/home/q/opensmile/human_result.json') as rf:
    #with open("/home/q/ruixiong.zhang/emotion_file/emotion_data/test_data.json", 'r') as f, open('/home/q/opensmile/test_result.json') as rf:
        for egmap in rf:
            egmap = json.loads(egmap)
            feat_map[egmap['key'].encode('utf-8')] = egmap['result']
        
        lines = f.readlines()
        for index, line in enumerate(lines):
            content = json.loads(line)
            result = []
            if index < test_line:
                train_part = False
            else:
                train_part = True
            if not train_part:
                result.append(content['key'])
            try:
                result.extend(feat_map[content['key'].encode('utf-8')])
            except:
                traceback.print_exc()
                continue
            if train_part:
                X_train.append(result)
                if int(content['type']) in [2, 3]:
                    Y_train.append(2)
                else:
                    Y_train.append(int(content['type'])) 
            else:
                x_test.append(result)
                if int(content['type']) in [2, 3]:
                    y_test.append(2)
                else:
                    y_test.append(int(content['type'])) 
                #y_test.append(0)
            
    #print 'X_train:', len(X_train), len(X_train[0]), 'Y_train:', len(filter(lambda x: x == 1, Y_train))
    print len(X_train), len(x_test)
    return X_train, Y_train, x_test, y_test
