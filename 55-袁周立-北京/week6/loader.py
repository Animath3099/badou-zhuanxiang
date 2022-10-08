# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.utils.data import random_split
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '好评', 1: '差评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        data = np.loadtxt(self.path, encoding="utf-8", delimiter="\n", dtype=str, skiprows=1)
        for line in data:
            label, content = line.split(",", 1)
            if label.isdigit():
                input_id = self.encode_sentence(content)
                input_id = torch.LongTensor(input_id)
                label = torch.LongTensor([int(label)])
                self.data.append([input_id, label])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, k, shuffle=True):
    dg = DataGenerator(data_path, config)
    # dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)

    train_data, test_data = random_split(dataset=dg, lengths=[len(dg) - int(len(dg) / k), int(len(dg) / k)],
                                         generator=torch.Generator().manual_seed(config["seed"]))
    train_data = DataLoader(train_data, batch_size=config["batch_size"], shuffle=shuffle)
    test_data = DataLoader(test_data, batch_size=config["batch_size"], shuffle=shuffle)
    return [train_data, test_data]

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("文本分类练习.csv", Config)
    print(dg[1])
