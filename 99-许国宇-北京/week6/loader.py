import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
"""
loading the data
"""

class DataGenerator:
    def __init__(self,data_path,config):
        self.config=config
        self.path=data_path
        self.index_to_label={0:"差评",1:"好评"}
        self.label_to_index=dict((y,x) for x,y in self.index_to_label.items())
        self.config["class_num"]=len(self.index_to_label)
        if self.config["model_type"]=="bert":
            self.tokenizer=BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab=load_vocab(config["vocab_path"])
        self.config["vocab_size"]=len(self.vocab)
        self.load()
    def load(self):
        self.data=[]
        data=pd.read_csv(self.path)
        for label,review in zip(data["label"],data["review"]):
            if self.config["model_type"]=="bert":
                input_id=self.tokenizer.encode(review,max_length=self.config["max_length"],pad_to_max_length=True)
            else:
                input_id=self.encode_sentence(review)
            input_id=torch.LongTensor(input_id)
            label_index=torch.LongTensor([label])
            self.data.append([input_id,label_index])
        return
    def encode_sentence(self,text):
        input_id=[]
        for char in text:
            input_id.append(self.vocab.get(char,self.vocab["[UNK]"]))
        input_id=self.padding(input_id)
        return input_id
    #completion or truncation the input sequence,make it possible to calulate in a batch
    def padding(self,input_id):
        input_id=input_id[:self.config["max_length"]]
        input_id+=[0]*(self.config["max_length"]-len(input_id))
        return input_id
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    token_dict={}
    with open(vocab_path,encoding="utf8") as f:
        for index,line in enumerate(f):
            token=line.strip()
            token_dict[token]=index+1 #the position 0 is retained for padding,so the number start from 1.
    return token_dict

#encapsulate the data with the dataloader class that comes with torch
def load_data(data_path,config,shuffle=True):
    dg=DataGenerator(data_path,config)
    dl=DataLoader(dg,batch_size=config["batch_size"],shuffle=shuffle)
    return dl

if __name__=="__main__":
    from config import Config
    dg=DataGenerator("valid_tag_news.json",Config)
    print(dg[1])