#!usr/bin/env python
#-*- coding:utf-8 -*-

import os

def load_toutiao_dataset1():
    dataset = []
    file_path = 'data/toutiao_cat_data.txt'
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split('_!_')[3]
            dataset.append(line)
    return dataset

def load_toutiao_dataset2():
    dataset = []
    file_path = 'data/nlp7294/'
    for file in os.listdir(file_path):
        index = -1
        with open(file_path+file, 'r') as f:
            for line in f:
                index += 1
                if index == 0:
                    continue
                line = line.strip().split()[1]
                dataset.append(line)
    return dataset


def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            dataset.append(line)
    return dataset

def save_data(dataset, fileName):
    with open(fileName, 'w') as f:
        for line in dataset:
            f.write(line+'\n')
    print('Done')

if __name__ == "__main__":
    dataset1 = load_toutiao_dataset1()
    print(len(dataset1))
    print(len(set(dataset1)))
    print(dataset1[0])
    dataset2 = load_toutiao_dataset2()
    print(len(dataset2))
    print(len(set(dataset2)))
    print(dataset2[0])
    dataset = dataset1 + dataset2
    print(len(dataset))
    print(len(set(dataset)))


    save_data(dataset, 'data/processed_data/all_same_765376/all_data_765376.txt')
    save_data(list(set(dataset)), 'data/processed_data/no_same_330748/nosame_data_330748.txt')

