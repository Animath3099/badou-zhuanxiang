import random

import numpy as np
import torch.nn as nn
import torch.nn.functional
import matplotlib.pyplot as plt

"""
多分类任务：
包含 123456789 进行分类
包含 1 是第一类
包含 2 是第二类
包含多个数字，数字小的优先级高
词典：26个字母加9个数字
没有数字是第十类

每个文本7个字符
每个字符嵌入为20维向量
"""

class MyModel(nn.Module):
    def __init__(self, vocab_size, word_dim, sentence_len):

        super(MyModel, self).__init__()

        # 词嵌入，每个字符转为word_dim维向量
        self.embedding = nn.Embedding(vocab_size, word_dim)
        # 池化
        self.pool = nn.AvgPool1d(sentence_len)
        # 线性化
        self.linear = nn.Linear(word_dim, 10)
        # 激活函数
        self.activate = torch.softmax
        # 损失函数
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)   # (batch_size,7,1) -> (batch_size,7,20)
        x = self.pool(x.transpose(1,2)).squeeze()    # 把7池化掉，相当于在竖着的方向上进行最大值池化
        x = self.linear(x)
        y_pred = self.activate(x, dim=1)
        if y is None:
            return y_pred
        else:
            return self.loss(y, y_pred)


# 随机生成一个文本
def get_rand_simple(vocab, sentence_len):
    x = [random.choice(list(vocab.keys())) for i in range(sentence_len)]
    if "1" in x:
        y = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif "2" in x:
        y = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif "3" in x:
        y = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif "4" in x:
        y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif "5" in x:
        y = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif "6" in x:
        y = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif "7" in x:
        y = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif "8" in x:
        y = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif "9" in x:
        y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    else:
        y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    x = [vocab[e] for e in x]
    return x, y

# 获取词典
def get_vocab():
    dic = {}
    for index, e in enumerate("abcdefghijklmnopqrstuvwxyz123456789"):
        dic[e] = index
    dic["undefined"] = len(dic)
    return dic


# 生成一个batch的样本
def get_batch_simple(batch_size, vocab, sentence_len):
    x = []
    y = []
    for i in range(batch_size):
        new_simple = get_rand_simple(vocab, sentence_len)
        x.append(new_simple[0])
        y.append(new_simple[1])
    return torch.LongTensor(x), torch.FloatTensor(y)

# 测试模型
def evaluate(model, vocab, sentence_len):
    model.eval()
    eval_x, eval_y = get_batch_simple(200, vocab, sentence_len)
    print("本轮测试集中十类样本的数量分别为：{}".format(torch.sum(eval_y, dim=0)))
    correct = 0
    wrong = 0
    with torch.no_grad():
        y = model(eval_x)
        for y_p, y_t in zip(y, eval_y):
            if y_t[torch.argmax(y_p)] == 1:
                correct += 1
            else:
                wrong += 1
    print("测试正确数量：{}, 准确率：{}".format(correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    word_dim = 20
    sentence_len = 7
    batch_size = 20
    epoch = 30
    train_simple = 1000
    learn_rate = 0.003

    vocab = get_vocab()
    my_model = MyModel(len(vocab), word_dim, sentence_len)
    optim = torch.optim.Adam(my_model.parameters(), lr=learn_rate)

    loss_data = []
    accs = []
    for e in range(epoch):
        my_model.train()
        cur_loss_data = []
        for i in range(int(train_simple / batch_size)):
            x, y = get_batch_simple(batch_size, vocab, sentence_len)

            optim.zero_grad()
            loss = my_model(x, y)
            loss.backward()
            optim.step()
            cur_loss_data.append(loss.item())

        print("第{}轮平均loss值：{}".format(e + 1, np.mean(cur_loss_data)))
        loss_data.append(np.mean(cur_loss_data))
        accs.append(evaluate(my_model, vocab, sentence_len))

    plt.plot(range(len(loss_data)), loss_data, label="loss")
    plt.plot(range(len(accs)), accs, label="acc")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
