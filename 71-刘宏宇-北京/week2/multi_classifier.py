import json
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    '''
    构造多分类模型
    '''
    def __init__(self, char_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), char_dim)  #embedding层
        self.pooling = nn.AvgPool1d(sentence_length)  #池化层
        self.classify = nn.Linear(char_dim, 5)  #线性层
        self.activation = torch.softmax

        self.loss = nn.CrossEntropyLoss()  #loss函数采用交叉熵损失, 将softmax归一化的过程和交叉熵评估概率的过程合并

    def forward(self, x, y=None):
        x = self.embedding(x)  #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = self.pooling(x.transpose(1,2)).squeeze() #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim)
        x = self.classify(x) # (batch_size, vector_dim) -> (batch_size, 5)

        if y is not None:
            return self.loss(x, y)
        else:
            return self.activation(x, 1)

def build_vocab():
    '''
    建立字表
    '''
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集

    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index
    vocab['unk'] = len(vocab)
    return vocab

def build_model(vocab, char_dim, sentence_length):
    '''
    建立模型
    '''
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

def build_sample(vocab, sentence_length):
    '''
    #构造一条训练样本
    '''

    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if set("mc") & set(x):
        y = 1
    elif set("iz") & set(x):
        y = 2
    elif set("gr") & set(x):
        y = 3
    elif set("bk") & set(x):
        y = 4
    else:
        y = 0
    
    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, y

def build_dataset(batch_size, vocab, sentence_length):
    '''
    #构造训练样本
    '''
    dataset_x = []; dataset_y = []
    for i in range(batch_size):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def evaluate(model, vocab, sentence_length, x, y):
    '''
    测试模型准确率
    '''
    model.eval()

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x).detach().numpy()

        for y_p, y_t in zip(y_pred, y):
            y_p = list(y_p)
            if y_p.index(max(y_p)) == y_t:
                correct += 1
            else:
                wrong += 1

    print("样本总数：%d, 正确预测个数：%d, 正确率：%f"%(len(y.numpy()), correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():

    #配置参数
    epoch_num = 10       #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 1000    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 10   #样本文本长度
    learning_rate = 0.005 #学习率

    test_sample = 1000
    
    # 建立字表
    vocab = build_vocab()

    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 构建测试数据
    test_x, test_y = build_dataset(test_sample, vocab, sentence_length)

    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()

        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            #构造一组训练样本
            x, y = build_dataset(batch_size, vocab, sentence_length)

            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()

            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        acc = evaluate(model, vocab, sentence_length, test_x, test_y)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 画图
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    # plt.legend()
    # plt.show()

    # 保存模型
    torch.save(model.state_dict(), "multi_classify_model.pth")

    # 保存词表
    writer = open("m_vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()

def predict(model_path, vocab_path, input_strings):
    char_dim = 20         
    sentence_length = 10   
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))

    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))

    dataset_x = []
    for input_string in input_strings:
        x = [vocab.get(char, vocab['unk']) for char in input_string]
        dataset_x.append(x)

    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(dataset_x))

    for i, input_string in enumerate(input_strings):
        res = list(result[i].detach().numpy())
        print(res)

        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, res.index(max(res)), max(res))) #打印结果


if __name__ == "__main__":
    # main()

    test_strings = ["ffvaeevaee", "cwsdfgsdfg", "rqwdygwdyg", "nlkwwwkwww"]
    predict("multi_classify_model.pth", "vocab.json", test_strings)