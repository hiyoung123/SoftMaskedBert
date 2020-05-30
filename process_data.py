#!usr/bin/env python
#-*- coding:utf-8 -*-

import random
from pypinyin import lazy_pinyin
from data_loader import load_dataset, save_data

def is_china_char(ch):
    if u'\u4e00' <= ch <= u'\u9fff':
        return True
    return False

def gen_char_dict(dataset):
    char_dict = {}
    for line in dataset:
        line = line.strip()
        for char in line:
            if len(char) != 0 and is_china_char(char):
                char_dict[char] = char_dict.get(char, 0) + 1
    return char_dict

def gen_pinyin_dict(dataset, char_dict):
    char_pinyin_dict = {}
    for i, line in enumerate(dataset):
        print(i)
        line = line.strip()
        for c in line:
            if is_china_char(c):
                word_pinyin = ''.join(lazy_pinyin(c))
                if len(word_pinyin) != 0:
                    if word_pinyin not in char_pinyin_dict:
                        char_pinyin_dict[word_pinyin] = c + '_' + str(char_dict.get(c, 0))
                    else:
                        char_pinyin_dict[word_pinyin] += ';' + c + '_' + str(char_dict.get(c, 0))

    data = {}
    for pinyin, words in char_pinyin_dict.items():
        tmp = {}
        for word in words.split(';'):
            print(word)
            if len(word) != 0:
                word_word = word.split('_')[0]
                word_count = int(word.split('_')[1])
                tmp[word_word] = word_count
        data[pinyin] = tmp


    f = open('data/pinyin2char.model', 'w')
    f.write(str(data))
    f.close()

def load_pinyin_dict(file_path):
    f = open(file_path, 'r')
    a = f.read()
    char_pinyin_dict = eval(a)
    f.close()
    return char_pinyin_dict


def random_word(sentence, char_pinyin_dict, char_dict, confusion_dict=None):
    tokens = [x for x in sentence]
    for i, token in enumerate(sentence):
        if not is_china_char(token):
            continue
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15
            # 谐音 65%
            if prob < 0.85:
                candiation = char_pinyin_dict.get(''.join(lazy_pinyin(token)), {token:''})
                candiation = sorted(candiation.items(), key=lambda x:x[1], reverse=True)
                candiation = candiation[:int(len(candiation)/2+0.5)]
                candiation = [x[0] for x in candiation]
                tokens[i] = random.choice(candiation)
            # 随机 10%
            elif prob < 0.9:
                candiation = sorted(char_dict.items(), key=lambda x:x[1], reverse=True)
                candiation = candiation[:int(len(candiation)/2+0.5)]
                candiation = [x[0] for x in candiation]
                tokens[i] = random.choice(candiation)
            # 删除 5%
            elif prob < 0.95:
                tokens[i] = ''
            # 添加 5%
            else:
                candiation = sorted(char_dict.items(), key=lambda x:x[1], reverse=True)
                candiation = candiation[:int(len(candiation)/2+0.5)]
                candiation = [x[0] for x in candiation]
                tokens.insert(i+1, random.choice(candiation))
    return ''.join(tokens)


def random_dataset(dataset, char_pinyin_dict, char_dict):
    out = []
    for ids, line in enumerate(dataset):
        print(ids)
        line = random_word(line, char_pinyin_dict, char_dict)
        out.append(line)
    return out


if __name__ == '__main__':
    dataset = load_dataset('data/processed_data/all_data_765376.txt')
    char_dict = gen_char_dict(dataset)
    # gen_pinyin_dict(dataset, char_dict)
    char_pinyin_dict = load_pinyin_dict('data/pinyin2char.model')
    process_dataset = random_dataset(dataset, char_pinyin_dict, char_dict)
    save_data(process_dataset, 'data/processed_data/process_data_765376.txt')
