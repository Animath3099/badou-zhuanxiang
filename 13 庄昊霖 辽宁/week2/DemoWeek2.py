# coding:utf8

"""
0.创建字表
1.创建模型
2.选择优化器
3.创建训练集
4.训练
5.测试每轮模型的准确率
6.用训练的模型做预测
"""
import torch.nn as nn
import torch
import random
import numpy as np
import json
import math


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        self.pool = nn.AvgPool1d(sentence_length)
        self.activation = torch.sigmoid
        self.ce_classify = nn.Linear(vector_dim, 3)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)
        x = self.pool(x.transpose(1, 2)).squeeze()
        x = self.ce_classify(x)
        y_pred = self.activation(x)

        if y is not None:
            return self.ce_loss(y_pred, y)   # 预测值和真实值计算损失
        else:
            return y_pred                 # 输出预测结果


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 指定哪些字出现时为正样本：a或b或c出现
    if set("abc") & set(x):
        y = 1
    elif set('xyz') & set(x):
        y = 2
    # 指定字都未出现，则为负样本
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]   # 将字转换成序号，为了做embedding
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim, sentence_length):
    """params: 字表  字符转化成的维度  字符长度"""
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


def main():
    # 配置参数
    epoch_num = 10         # 训练轮数
    batch_size = 20        # 每次训练样本个数
    train_sample = 500     # 每轮训练总共训练的样本总数
    char_dim = 20          # 每个字的维度
    sentence_length = 6    # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 1.建立字表
    vocab = build_vocab()
    # 2.建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 3.选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 4.训练过程---多轮训练--写法固定
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()    # 梯度归零
            loss = model(x, y)   # 计算loss：当前预测值和真实值之间的差距
            loss.backward()      # 计算梯度：反向传播
            optim.step()         # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    torch.save(model.state_dict(), "model3.pth")
    return


def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)      # 建立模型
    model.load_state_dict(torch.load(model_path))              # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()   # 测试模式 不会使用dropout
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        result_pre = np.array(result[i])
        result_pre = int(np.argmax(result_pre))

        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, result_pre, result[i][result_pre]))  # 打印结果


if __name__ == '__main__':
    # main()
    test_strings = ["ffvaee", "pwsdfg", "rqwdlg", "nlxwww"]
    predict("model3.pth", "vocab.json", test_strings)
