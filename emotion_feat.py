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
import sys

import soundfile
import numpy as np
from numpy.lib.stride_tricks import as_strided
from pydub import AudioSegment


y_map = {'angry':0, 'sad':1, 'happy':2, 'fear':3, 'neutral':4, 'surprise':5}


def create_feats():
    X_train, Y_train, x_val, y_val = [], [], [], []
    train_part = False
    val_line = 190
    feat_map = {}
    with open("casia/emotion_shuffle.json", 'r') as f, open('/home/q/opensmile/caisa_result.json') as rf:
        for egmap in rf:
            egmap = json.loads(egmap)
            feat_map[egmap['key'].encode('utf-8')] = egmap['result']
        
        lines = f.readlines()
        
        for index, line in enumerate(lines):
            content = json.loads(line)
            result = []
            if index < val_line:
                train_part = False
            else:
                train_part = True
            if not train_part:
                # id
                result.append(content['key'])
            
            result.extend(feat_map[content['key'].encode('utf-8')])
            
            if train_part:
                X_train.append(result)
                Y_train.append(y_map[content['emotion']]) 
            else:
                x_val.append(result)
                y_val.append(y_map[content['emotion']]) 
            
    #print 'X_train:', len(X_train), len(X_train[0]), 'Y_train:', len(filter(lambda x: x == 1, Y_train))
    return X_train, Y_train, x_val, y_val
