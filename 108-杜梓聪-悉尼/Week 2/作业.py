#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 19:28:18 2022

@author: du
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
# import matplotlib.pyplot as plt

"""
Week 2 作业:
将课上demo的例子改造成多分类任务
例如字符串包含“abc”属于第一类，包含“xyz”属于第二类，其余属于第三类
修改模型结构和训练代码
完成训练
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #nn.Embedding方法可以将字母映射成向量（26*20）；embedding层
        self.pool = nn.AvgPool1d(sentence_length)   #池化层，例子里sentence_length = 4，代表在4那一维做平均，即每4个数求一下平均
        self.classify = nn.Linear(vector_dim, 3)     #线性层，y=kx+b，直接封装好k和b
        self.activation = nn.Sigmoid()  # 使用sigmoid，将全连接层的输出的3个神经元的值转化到01的概率上
        self.loss = nn.CrossEntropyLoss()  #loss函数采用交叉熵损失,无单独求需softmax归一化后的值，因为nn.CrossEntropyLoss()会自动先帮你求softmax

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)；即输入有20句话，每句话由6个单词组成，输出变成20句话*6个词*词向量纬度20；
        x = self.pool(x.transpose(1, 2)).squeeze() #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim)；pooling操作默认对于输入张量的最后一维进行，因此需要把平均的那维transpose到最后，所以sen_len和vector_dim位置要互换；平均完后变成1*5了，用squeeze可以去掉维数为1的纬度；
        x = self.classify(x)                       #(batch_size, vector_dim) -> (batch_size, 3)
        y_pred = self.activation(x)                #(batch_size, 1) -> (batch_size, 1)使用sigmoid，将全连接层的输出的3个神经元的值转化到01的概率上
        # 在使用nn.CrossEntropyLoss损失函数时，它要求传入的target必须是1D，像这样：[1,2,3,4,5]，这样[[1],[2],[3]]是不行的！！
        # 同时，使用CrossEntropyLoss时，要求第一个参数为网络输出值，FloatTensor类型，第二个参数为目标值，LongTensor类型。
        if y is not None: # 如果传y进来，就代表需要求loss，否则直接输出预测值y_pred
            return self.loss(y_pred, y.squeeze().long())  #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

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
def build_sample(vocab, sentence_length): # 生成一条样本
    #随机从字表选取sentence_length (6)个字母，可能重复：['n', 'm', 'q', 'x', 'g', 'n']
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    a = 0
    b = 0
    c = 0
    #指定哪些字出现时为正样本
    if set("abc") & set(x):
        y = int(0)
        a = 1
    elif set("xyz") & set(x):
        y = int(1)
        b = 1
    else:
        y = int(2)
        c = 1
    # 将字转换成序号，为了做embedding
    # 字典D.get()函数用于返回指定键的值，如果值不在字典中返回默认值。vocab.get('a')=0,vocab.get('2')=26，即返回'unk'的值
    x = [vocab.get(word, vocab['unk']) for word in x]   
    return x, y, a, b, c
    # 以生成第二类样本为例：return [5, 21, 15, 25, 23, 4] 2 0 1 0

#建立数据集
#输入需要的样本数量。需要多少生成多少
# sample_length = 20, sentence_length = 6。每个batch都重新单独建立20句话的测试样本，每句话6个单词
def build_dataset(sample_length, vocab, sentence_length):
    count_1 = 0
    count_2 = 0
    count_3 = 0
    dataset_x = []
    dataset_y = []
    for i in range(sample_length): # 要生成20句话，需要重复20轮的生成步骤
        x, y, a, b, c = build_sample(vocab, sentence_length) # 生成1句话
        dataset_x.append(x)
        dataset_y.append([y])
        count_1 += a
        count_2 += b
        count_3 += c
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y),count_1, count_2, count_3

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y, count_1, count_2, count_3 = build_dataset(200, vocab, sample_length) #建立200条用于测试的样本
    
    print("本次测试集中共有%d个一类样本，%d个二类样本，%d个三类样本"%(count_1, count_2, count_3))
    correct, wrong = 0, 0
    with torch.no_grad(): # 不需要反向传播
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            y_p = y_p.tolist() # 将Tensor转为List，再取出list里最大的那个值的索引，索引值即为预测的类别
            if y_p.index(max(y_p)) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    #配置参数
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #一个epoch总共训练的样本总数是500，每个batch=20，所以一个epoch需要训练500/20=25个iteration
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num): # 共训练10个epochs
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)): # 500/20=25个iteration
            x, y, _, _, _ = build_dataset(batch_size, vocab, sentence_length) #重新去构造一个batch的训练样本，一个batch共20句话
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # #画图
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    # plt.legend()
    # plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        temp = result[i].tolist()   # 需要把结果转为列表，取最大值的位置作为预测的类别
        ans = temp.index(max(temp))
        print("输入：%s, 预测类别：%d" % (input_string, ans)) #打印结果



if __name__ == "__main__":
    main()
    test_strings = ["ffvaee", "cwsdfg", "rqwdyg", "nlkwww"]
    print("\n测试：")
    predict("model.pth", "vocab.json", test_strings)
    
    
    
    
    
    
    