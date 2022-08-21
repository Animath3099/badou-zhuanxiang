import random
import numpy as np
import torch
import torch.nn as nn


class Torchmodel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(Torchmodel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        self.rnn = nn.RNN(num_layers=1, input_size=vector_dim, hidden_size=vector_dim, batch_first=True)
        self.pool = nn.AvgPool1d(sentence_length)
        self.classify = nn.Linear(vector_dim, 3)
        self.sigmoid = torch.sigmoid
        self.dp = torch.nn.Dropout(0.4)
        # self.loss = nn.functional.mse_loss
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        _, x = self.rnn(x)
        # x = self.pool(x.transpose(1,2)).squeeze()
        # x = self.classify(x)
        y_pred = self.classify(x.squeeze())
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if set('abc') & set(x):
        y = 0
    elif set('def') & set(x):
        y = 1
    else:
        y = 2
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    # return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    print('本次测试集中共有%d个0类样本，%d个1类样本，%d个2类样本' % (sum(y.eq(0)), sum(y.eq(1)), sum(y.eq(2))))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim=-1)
        correct += int(sum(y_pred == y.squeeze()))
        wrong += len(y) - correct
        # for y_p, y_t in zip(y_pred, y):
        #     if float(y_p) < 0.5 and int(y_t) == 0:
        #         correct += 1
        #     elif float(y_p) >= 0.5 and int(y_t) == 1:
        #         correct += 1
        #     else:
        #         wrong += 1
    print('正确预测个数：%d，正确率：%f' % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 20              # 训练轮数
    batch_size = 20             # 每次训练样本个数
    train_sample = 500          # 每轮训练总共训练的样本总数
    char_dim = 20               # 每个字的维度
    sentence_length = 6         # 样本文本长度
    learning_rate = 0.005       # 学习率

    vocab = build_vocab()
    model = Torchmodel(char_dim, sentence_length, vocab)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)      # 优化器
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    return


if __name__ == '__main__':
    main()
