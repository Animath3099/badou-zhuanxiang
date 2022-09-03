import numpy as np
import torch
import torch.nn as nn
import random
import json
import matplotlib.pyplot as plt
class TorchModel(nn.Module):
    def __init__(self,vocab,sentence_length,char_dim):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab),char_dim)
        self.pool = nn.AvgPool1d(sentence_length)
        self.classify = nn.Linear(char_dim,2)
        self.activation = torch.softmax
        self.loss = nn.functional.cross_entropy
    def forward(self,x,y=None):
        x = self.embedding(x)
        x = self.pool(x.transpose(1,2)).squeeze()
        #test_x = x
        x = self.classify(x)
        x = torch.FloatTensor(x)
        y_pred = self.activation(x,dim=1)
        #print("TorchModel y_pred:%f target_y:%f\n",y_pred,y)
        if y is not None:
            y = torch.Tensor(y).float()
            return self.loss(y,y_pred)
        else:
            return y_pred

# class DiyModel(nn.Module): ##用来验证TorchModel的前向传播是否正确
#     def __init__(self,w,b):
#         super(DiyModel,self).__init__()
#         self.w = w
#         self.b = b
#     def forward(self,x,y=None):
#         liner = np.dot(x,self.w.T) + self.b.T
#         y_pred = []
#         for line in liner:
#             sumall = np.sum(np.exp(line))
#             tmp = []
#             for term in line:
#                 tmp.append(np.exp(term)/sumall)
#             y_pred.append(tmp)
#         #print("DiyModel y_pred:",y_pred)
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    indexs = np.arange(len(chars))
    vocab = {}

    for index,char in zip(indexs,chars):
        vocab[char] = index
    vocab['unk'] = len(chars)
    return vocab

def build_model(vocab,sentence_length,char_dim):
    model = TorchModel(vocab,sentence_length,char_dim)
    return model

def build_sample(vocab,sentence_lengtgh):
    #random.choice(seq)
    #np.random.choice(val)
    x = [random.choice(list(vocab.values())) for i in range(sentence_lengtgh)]
    #print(x)
    if set([0,1,2]) & set(x):
        y = [0,1]
    else:
        y = [1,0]
    return x,y

def build_dataset(sample_num,vocab,setence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_num):
        x,y = build_sample(vocab,setence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x),torch.LongTensor(dataset_y)

def evaluate(model, vocab, sample_length):
    model.eval()
    x,y = build_dataset(200,vocab,sample_length)
    print("本次测试正样本%d个，负样本%d个"%(torch.sum(y,dim=0)[0], 200 - torch.sum(y,dim=0)[0]))

    correct,wrong = 0,0
    with torch.no_grad():
        y_pred = model.forward(x)
        for y_p,y_t in zip(y_pred,y):
            if float(y_p[0])>=0.5 and int(y_t[0])==1:
                correct+=1
            elif float(y_p[0])<0.5 and int(y_t[1])==1:
                correct+=1
            else:
                wrong+=1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct/(correct+wrong)
def main():

    #参数配置
    epoch_num = 10
    batch_size = 20
    train_sample = 500
    sentence_length = 6
    char_dim = 20
    learning_rate = 0.01

    vocab = build_vocab()
    dataset_x,dataset_y = build_dataset(train_sample,vocab, sentence_length)
    #print(dataset_x,dataset_y)
    model = build_model(vocab,sentence_length,char_dim)
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    log = []
    Acc = []
    print()
    for epoc in range(epoch_num):
        model.train()
        watch_loss = []
        for bath in range(train_sample//batch_size):
            x,y = build_dataset(batch_size,vocab,sentence_length)
            optim.zero_grad()
            loss = model.forward(x,y)
            #print(loss)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("第%d轮平均损失函数:%f"%(epoc+1,np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append(np.mean(watch_loss))
        Acc.append(acc)


    plt.plot(np.arange(epoch_num),log)
    plt.plot(np.arange(epoch_num), Acc)
    plt.show()

    torch.save(model.state_dict(),'Multimodel.pth')

    writer = open('vocab2.json','w',encoding='utf-8')
    return

def predict(model_pth,chars):

    setence_length = 6
    char_dim = 20
    vocab = build_vocab()
    model = build_model(vocab,setence_length,char_dim)
    model.load_state_dict(torch.load(model_pth))
    x = []
    for word in chars:
        seq = [vocab[letter] for letter in word]
        x.append(seq)
    x = torch.LongTensor(x)
    print(x)
    with torch.no_grad():
        model.eval()
        y_pred = model.forward(x)
        for index,y_p in enumerate(y_pred):
            if float(y_p[0])>=0.5:
                y_p[0] = 1
                y_p[1] = 0
            else:
                y_p[0] = 0
                y_p[1] = 1
            print("Inpust String:",chars[index],",class:",y_p)
        print("Inpust String:", chars, ",class:", y_pred)

if __name__ == "__main__":
    #main()
    test_strings = ["ffvaee", "cwsdfg", "rqwdyg", "nlkwww"]
    predict("Multimodel.pth", test_strings)
