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
    X_train, Y_train, x_val, y_val = [], [], [], []
    train_part = False
    val_line = 394675
    with open("resume_data/featY_newposi.txt", 'r') as f:
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
                result.append(content['id'])
            # gender
            result.append(int(content['gender']))
            # age
            age = int(content['age'])
            if age > 50 or age < 18:
                age = -1
            result.append(age)
            # educationDegree
            result.append(int(content['educationDegree']))
            # educationSchool
            result.append(int(content['educationSchool']))      
            # educationMajor
            result.append(int(content['educationMajor']))
            # descNum
            result.append(int(content['descNum']))
            # # last 10 desc
            # descCounts = content['descCounts']
            # if not descCounts:
            #     descCounts = [-1] * 10
            # descCounts = descCounts[-10:]
            # if len(descCounts) < 10:
            #     count = 10 - len(descCounts)
            #     descCounts = [-1] * count + descCounts
            # result.extend(descCounts)
            
            wordCount = 40
            descWords = content['descWords']
            if not descWords:
                descWords = [-1] * wordCount*2
            descWords = descWords[-wordCount*2:]
            if len(descWords) < wordCount*2:
                count = wordCount*2 - len(descWords)
                descWords = [-1] * count + descWords
            result.extend(descWords)   
            # expNum
            result.append(int(content['expNum'])) 
            # # last 10 exp
            # expCounts = content['expCounts']
            # if not expCounts:
            #     expCounts = [-1] * 10
            # expCounts = expCounts[-10:]
            # if len(expCounts) < 10:
            #     count = 10 - len(expCounts)
            #     expCounts = [-1] * count + expCounts
            # result.extend(expCounts)  
            
            expWords = content['expWords']
            if not expWords:
                expWords = [-1] * wordCount*2
            expWords = expWords[-wordCount*2:]
            if len(expWords) < wordCount*2:
                count = wordCount*2 - len(expWords)
                expWords = [-1] * count + expWords
            result.extend(expWords)   
            
            if train_part:
                X_train.append(result)
                Y_train.append(int(content['y'])) 
            else:
                x_val.append(result)
                y_val.append(int(content['y'])) 
            
    #print 'X_train:', len(X_train), len(X_train[0]), 'Y_train:', len(filter(lambda x: x == 1, Y_train))
    return X_train, Y_train, x_val, y_val
