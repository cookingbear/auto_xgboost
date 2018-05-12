# encoding: utf-8

'''
@author: cookingbear

@contact: 664610407@qq.com

@file: config_util.py

@time: 2018/5/12 下午5:41

'''

from collections import defaultdict


def get_config():
    config_dict = defaultdict(str)
    with open("config",'r') as f:
        for line in f:
            if '=' not in line:
                continue
            if line.startswith('#'):
                continue
            key, value = line.split('=')
            config_dict[key] = value.strip('\n').strip()
    return config_dict


def get_value(key):
    config_dict = get_config()
    return config_dict[key]


if __name__ == '__main__':
    print get_value('a')
