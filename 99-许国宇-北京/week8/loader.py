import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset,DataLoader

"""
Load the data
"""
#load word dict
def load_vocab(vocab_path):
    token_dict={}
    with open(vocab_path,encoding="utf8") as f:
        for index,line in enumerate(f):
            token=line.strip()
            #o is retained for padding so the position from 1 begin
            token_dict[token]=index+1
    return token_dict

class DataGenerator:
    def __init__(self,data_path,config):
        self.config=config
        self.path=data_path
        self.vocab=load_vocab(config["vocab_path"])
        self.config["vocab_size"]=len(self.vocab)
        self.sentences=[]
        self.schema=self.load_schema(config["schema_path"])
        self.load()
    def load(self):
        self.data=[]
        if list(self.path.split("\\"))[-1]=="usr_sentence.txt":
            with open(self.path, mode="r") as f:
                usr_question = f.readline()
                input_id = self.encode_sentence(usr_question)
                self.data.append(input_id)
            return input_id
        else:
            with open(self.path, encoding="utf8") as f:
                segments=f.read().split("\n\n")
                for segment in segments:
                    sentence=[]
                    labels=[]
                    for line in segment.split("\n"):
                        if line.strip()=="":
                            continue
                        char,label=line.split()
                        sentence.append(char)
                        labels.append(self.schema[label])
                    self.sentences.append("".join(sentence))
                    input_ids=self.encode_sentence(sentence)
                    labels=self.padding(labels,-1)
                    self.data.append([torch.LongTensor(input_ids),torch.LongTensor(labels)])
        return
    def encode_sentence(self,text,padding=True):
        input_id=[]
        if self.config["vocab_path"]=="words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word,self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char,self.vocab["[UNK]"]))
        if padding:
            input_id=self.padding(input_id)
        return input_id
    def padding(self,input_id,pad_token=0):
        input_id=input_id[:self.config["max_length"]]
        input_id+=[pad_token]*(self.config["max_length"]-len(input_id))
        return input_id
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    def load_schema(self,path):
        with open(path,encoding="utf8") as f:
            return json.load(f)

def load_data(data_path,config,shuffle=True):
    dg=DataGenerator(data_path,config)
    dl=DataLoader(dg,batch_size=config["batch_size"],shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)


