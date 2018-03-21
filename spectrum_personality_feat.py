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


def create_feats():
    X, Y = [], []
    with open("../data/personality_correct.json", 'r') as f:
        lines = f.readlines()
        
        for index, line in enumerate(lines):
            content = json.loads(line)
            result = []
            # score
            if not content['score']:
                print 'score_empty'
                continue
            result.append(float(content['score']))
            # leftDuration
            if not content['leftDuration']:
                print 'leftDuration_empty'
                continue
            #result.append(float(content['leftDuration']))
            # rightDuration
            if not content['rightDuration']:
                print 'rightDuration_empty'
                continue
            #result.append(float(content['rightDuration']))
            # duration
            if not content['during']:
                print 'duration_empty'
                continue
            result.append(float(content['during']))      
            # personality
            if not content['personality'] or content['personality'] == 'NULL' or content['personality'] == 'None':
                continue            
            personality_map = {'kq':1, 'gz':2, 'mty':3, 'ly':4}
            #result.append(personality_map[content['personality']])
            # hangupmode
            if not content['hangupMode']:
                print 'hangupMode_empty'
                continue                
            if content['hangupMode'] == 'R':
                hangupMode = 1
            else:
                hangupMode = 2
            result.append(hangupMode)
            
            X.append(result)
            if content['orderStatus'] == 1 or content['orderStatus'] == 2:
                Y.append(1)
            else:
                Y.append(0)
            
    print 'X:', len(X), 'Y:', len(filter(lambda x: x == 1, Y))
    return X,Y
