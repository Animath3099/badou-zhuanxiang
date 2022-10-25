import torch
import torch.nn as nn
from torch.optim import Adam, SGD


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layers = nn.Sequential(
            nn.RNN(input_size=hidden_size,
                   hidden_size=hidden_size,
                   num_layers=2,
                   batch_first=True,
                   bidirectional=True,
                   dropout=0.1),
            GetFirst(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.layers(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x

class GetFirst(nn.Module):
    def __init__(self):
        super(GetFirst, self).__init__()

    def forward(self, x):
        return x[0]

class SentenceMatch(nn.Module):
    def __init__(self, config):
        super(SentenceMatch, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.config = config
        self.loss = self.triplet_loss

    def cos_distance(self, a, b):
        # 将某一个维度除以那个维度对应的范数(默认是2范数)。
        a = nn.functional.normalize(a.float(), p=2, dim=-1)
        b = nn.functional.normalize(b.float(), p=2, dim=-1)
        # 矩阵点乘后对每一个样本(也就是每一行)求和
        cos = torch.sum(torch.mul(a, b), dim=-1)
        return 1 - cos

    def triplet_loss(self, a, p, n, marin=0.1):
        ap = self.cos_distance(a, p)
        an = self.cos_distance(a, n)
        diff = ap - an + marin
        # 存在diff大于0的样本时，计算平均值作为loss，否则loss为0（这么写不会显示loss为nan）
        if diff.gt(0).any():
            return torch.mean(diff[diff.gt(0)])
        else:
            return torch.mean(0 * diff)

    def forward(self, x1, x2=None, x3=None):
        if x2 is not None:
            sentence1_vector = self.sentence_encoder(x1)
            sentence2_vector = self.sentence_encoder(x2)
            sentence3_vector = self.sentence_encoder(x3)
            if x3 is not None:
                # 训练
                return self.loss(sentence1_vector, sentence2_vector, sentence3_vector)
            else:
                return self.cos_distance(sentence1_vector, sentence2_vector)
        else:
            # 使用向量化能力
            return self.sentence_encoder(x1)

    def choose_optimizer(self):
        optimizer = self.config["optimizer"]
        learning_rate = self.config["learning_rate"]
        if optimizer == "adam":
            return Adam(self.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            return SGD(self.parameters(), lr=learning_rate)
