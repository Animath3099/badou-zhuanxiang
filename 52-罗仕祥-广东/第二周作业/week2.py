#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

作业：
将课上demo的例子改造成多分类任务
例如字符串包含“abc”属于第一类，包含“xyz”属于第二类，其余属于第三类
修改模型结构和训练代码
完成训练
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()  # 实现调用父类
        # embedding层 ，建立词向量层
        self.embedding = nn.Embedding(len(vocab), vector_dim)   # 用法：nn.Embedding(num_embeddings-词典长度，embedding_dim-向量维度)
        self.rnn = nn.RNN(num_layers=1, input_size=vector_dim, hidden_size=vector_dim, batch_first=True)    # num_layers=1隐藏层的个数,表示1个RNN堆叠, input_size输入特征的维度, hidden_size输出的维度
        self.pool = nn.AvgPool1d(sentence_length)   #池化层
        # 线性层
        self.classify = nn.Linear(in_features=vector_dim, out_features=3)   # in_features：输入的神经元个数， out_features：输出神经元个数 ，  bias=True ：是否包含偏置)
        # # 使用torch计算交叉熵
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # 到x中取对应的词向量   #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output,h_n = self.rnn(x)    #(batch_size, sen_len, vector_dim), (1, batch_size, vector_dim)
        y_pred = self.classify(h_n.squeeze())                       #(batch_size, vector_dim) -> (batch_size, 3)
        if y is not None:
            # 将tensorfloat转换为tensorlong: .long()
            return self.loss(y_pred, y.squeeze().long())   #预测值和真实值计算损失
        else:
            return y_pred.squeeze().long()                 #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index   #每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):

    # random.choice(sequence) 从序列中获取一个随机元素，sequence表示一个有序类型。
    # for _ in range(n) 语法中 _ 是占位符，只在乎遍历次数 range(n) 就是遍历n次
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)] # 随机从字表选取sentence_length个字，可能重复

    #指定哪些字出现时为正样本
    # set() 函数创建一个无序不重复元素集
    # x = set('runoob')     y = set('google')
    # x & y # 交集    x | y # 并集      x - y # 差集
    if set("abc") & set(x):
        y = 1
    elif set("xyz") & set(x):
        y = 2
    #指定字都未出现，则为负样本
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)     # 随机生成一个样本,并指出y是正样本还是负样本
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)    # torch.LongTensor是64位整型

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()    # 测试时开启
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    # torch.eq(input, other, *, out=None) 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False。
    # input ：必须是一个Tensor，该张量用于比较
    # other ：可以是一个张量Tensor，也可以是一个值value
    print("本次预测集中共有%d个1类样本，%d个2类样本，%d个其他类样本"%(sum(torch.eq(y,1)), sum(torch.eq(y,2)), sum(torch.eq(y,0))))
    correct, wrong = 0, 0
    with torch.no_grad():   # 数据不需要计算梯度，也不会进行反向传播
        y_pred = model(x)      #模型预测
        print(y_pred.shape)     # torch.Size([200, 3])
        y_pred = torch.argmax(y_pred, dim=-1)   # dim=-1表示张量维度的最低维度
        print(y_pred.shape)     # torch.Size([200])
        # print(y_pred, y.squeeze().long()))
        c = torch.eq(y_pred, y.squeeze().long()).numpy()    # 比较预测值与真实值中正确的情况,bool值
        correct = np.count_nonzero(c)   # 统计数组中真值的数量:np.count_nonzero(boolarr)     或np.sum(boolarr!=0)
        wrong += len(y) - correct
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%s" % (input_string, int(torch.argmax(result[i])), result[i]))   # 打印结果

def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)

    # 选择优化器,这里使用Adam优化算法
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()   # 训练时开启
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss    等价于 loss = model.forward(x, y)
            loss.backward()  # 反向传播求解梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())  # pytorch中的.item()用于将一个零维张量转换成浮点数
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()

if __name__ == '__main__':
    main()
    test_strings = ["fplwsx", "cdedfg", "ryhnmg", "nlkplt", 'awefxv', 'yyzhio']
    predict("model.pth", "vocab.json", test_strings)





