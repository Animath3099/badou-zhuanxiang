import json
import random
from collections import defaultdict
import torch
from torch.utils.data import DataLoader

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = self.load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.scheme_dict = self.load_schema(config["schema_path"])
        self.data_type = None
        self.load()

    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf-8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1   # 0留给padding位置，所以从1开始
        return token_dict

    def load_schema(self, schema_path):
        with open(schema_path, encoding="utf-8") as f:
            return json.loads(f.read())

    def load(self):
        self.knwb = defaultdict(list)
        self.test_data = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    target = line["target"]
                    for question in questions:
                        self.knwb[self.scheme_dict[target]].append(torch.LongTensor(self.encode_sentence(question)))
                else:
                    self.data_type = "test"
                    question, label = line
                    self.test_data.append([torch.LongTensor(self.encode_sentence(question)),
                                           torch.LongTensor([self.scheme_dict[label]])])

    def encode_sentence(self, sentence):
        char_indexs = []
        for char in sentence:
            char_indexs.append(self.vocab.get(char, self.vocab["[UNK]"]))
        char_indexs = self.padding(char_indexs)
        return char_indexs

    def padding(self, char_indexs):
        char_indexs = char_indexs[:self.config["max_length"]]
        char_indexs += [0] * (self.config["max_length"] - len(char_indexs))
        return char_indexs

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            return len(self.test_data)

    def __getitem__(self, item):
        if self.data_type == "train":
            return self.random_train_sample()
        else:
            return self.test_data[item]

    def get_knwb(self):
        return self.knwb

    def random_train_sample(self):
        total_target_indexs = list(self.knwb.keys())
        target_index1, target_index2 = random.sample(total_target_indexs, 2)
        if len(self.knwb[target_index1]) < 2:
            return self.random_train_sample()
        sample1, sample2 = random.sample(self.knwb[target_index1], 2)
        sample3 = random.choice(self.knwb[target_index2])
        return [sample1, sample2, sample3]


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
