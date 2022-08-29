import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TorchModel(nn.Module):

    def __init__(self, vocab, vocab_dim, sentence_length):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), vocab_dim)
        self.pool = nn.AvgPool1d(sentence_length)
        self.activation = torch.tanh
        self.loss = nn.CrossEntropyLoss()
        self.dp = nn.Dropout(0.15)
        self.hidden1 = nn.Linear(vocab_dim, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.hidden3 = nn.Linear(64, 4)
        self.classify = torch.softmax

    def forward(self, x, y=None):
        x = self.embedding(x)
        x = self.pool(x.transpose(1, 2)).squeeze()
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.dp(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.dp(x)
        x = self.hidden3(x)
        # x = self.activation(x)
        y_predict = self.classify(x, -1)
        if y is not None:
            return self.loss(y_predict, y)
        else:
            return y_predict


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if set("abc") & set(x):  # 交集检验
        y = 0
    elif set("jkl") & set(x):
        y = 1
    elif set("xyz") & set(x):
        y = 2
    else:
        y = 3
    # print(x, y)
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    # nn.functional.one_hot(torch.tensor(dataset_y), 7)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, vocab_dim, sentence_length):
    model = TorchModel(vocab, vocab_dim, sentence_length)
    return model


def evalute(model, vocab, sample_length):
    model.eval()  # model.eval() 作用等同于 self.train(False) 简而言之，就是评估模式。而非训练模式。
    x, y = build_dataset(200, vocab, sample_length)
    x = x.to(device)
    y = y.to(device)
    samples_classes = y.tolist()
    temp = set(samples_classes)
    class_count = {}
    for i in temp:
        class_count.update({i: samples_classes.count(i)})
    print(f"本次预测集中共有类别信息如下：{class_count}")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == y_t:
                # print(f"预测正确：预测分类{torch.argmax(y_p)},真实分类{y_t}")
                correct += 1
            else:
                # print(f"预测错误：预测分类{torch.argmax(y_p)},真实分类{y_t}")
                wrong += 1
    print(f"正确预测个数：{correct},正确率{correct/(correct+wrong)}")
    return correct/(correct+wrong)


def main():
    epoch = 10
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005

    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample/batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch+1}轮平均loss:{np.mean(watch_loss)}========")
        accuracy = evalute(model, vocab, sentence_length)
        log.append([accuracy, np.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log], label='accuracy')
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), 'myModel.pth')
    writer = open("myVocab.json", 'w', encoding='utf8')
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding='utf8'))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x =[]
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print(f"输入：{input_string}，预测类别：{torch.argmax(result[i], dim=0)}，概率值：{result[i][torch.argmax(result[i], dim=0)]}")


if __name__ == "__main__":
    main()
    # test_strings = ['ffvaee', 'cwsdfg', 'rqwdyg', '0acb23']
    # predict('myModel.pth', 'myVocab.json', test_strings)
