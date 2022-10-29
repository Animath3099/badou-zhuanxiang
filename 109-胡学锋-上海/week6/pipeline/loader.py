import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""
class DataGenerator:
    def __init__(self,data_path,config):
        #加载配置
        self.path =data_path
        self.config = config
        #设立有监督分类任务标签
        self.index_to_label = {1:'积极',0:'消极'}
        self.label_to_index = dict( (y,x) for x,y in self.index_to_label.items() )
        self.config["class_num"] = len(self.index_to_label)#习惯补全config信息
        #加载词典
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        #调用load方法，加载数据
        self.load()

    def load(self):
        self.data = []
        #读取语料
        with open(self.path,encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                label = int(line[0]) #标签
                text = line[2:] #文本内容

                input_id = self.encode_sentence(text) #拿去映射每个字的编号
                #转成tensor
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
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

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == '__main__':
    from config import Config
    dl =load_data(Config["train_data_path"],config=Config)
    dg = DataGenerator(Config["train_data_path"],config=Config)
    for index, batch_data in enumerate(dl):
        print(index)
        print(batch_data)
