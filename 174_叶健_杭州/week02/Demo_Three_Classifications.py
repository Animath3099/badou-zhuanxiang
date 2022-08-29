import random
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
import json
class TorchModel(nn.Module):
    def __init__(self,vector_dim, sentence_length, vocab):
        super(TorchModel,self).__init__()
        self.embedding = nn.Embedding(len(vocab),vector_dim) #embedding层
        self.pool = nn.AvgPool1d(sentence_length) #池化层
        self.classify1 = nn.Linear(vector_dim,3) #线性层
        # self.classify2 = nn.Linear(10,3) #线性层
        # self.activation=torch.sigmoid  #激活层归一化
        # self.loss=nn.functional.mse_loss #loss函数
        self.loss=nn.functional.cross_entropy #loss函数
    def forward(self,x,y=None):
        x = self.embedding(x) #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = self.pool(x.transpose(1,2)).squeeze() #(batch_size, sen_len, vector_dim) -> (batch_size,vector_dim)
        y_pred = self.classify1(x)  #(batch_size,vector_dim) -> (batch_size,1)
        # x = self.classify2(x)  #(batch_size,vector_dim) -> (batch_size,1)
        # y_pred = self.activation(x) #(batch_size,1) -> (batch_size,1)
        if y is None:
            return y_pred
        else:
            return self.loss(y_pred,y.squeeze())



#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars_list="abcdefghijklmnopqrstuvwxyz"
    vocab_dict={}
    for index,my_char in enumerate(chars_list):
        # print(index,my_char)
        vocab_dict[my_char]=index
    vocab_dict["unk"]=len(vocab_dict)
    return vocab_dict

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    x=[random.choice(list(vocab.keys())) for i in range(sentence_length)]
    # print(x)
    if (set("abc") & set(x)):
        y = 1
    elif (set("xyz") & set(x)):
        y = 2
    else:
        y = 0
    # print(x,y)
    x=[vocab.get(j,vocab['unk']) for j in x]
    return x,y
#
#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x=[]
    dataset_y=[]
    for word in range(sample_length):
        x,y=build_sample(vocab,sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    dataset_x=torch.LongTensor(dataset_x)
    dataset_y=torch.LongTensor(dataset_y)
    return dataset_x,dataset_y
#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    print("本次预测集中共有%d个0类，%d个1类，%d个二类样本"%(sum(y.eq(0)), sum(y.eq(1)), sum(y.eq(2))))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        # print(y_pred)
        # print("!~!~!~!!~")
        y_pred = torch.argmax(y_pred, dim=-1)
        # print(y_pred)
        # print(y.squeeze())
        # print(y_pred == y.squeeze())
        correct += int(sum(y_pred == y.squeeze()))
        wrong += len(y) - correct

    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)



def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    my_model = TorchModel(char_dim,sentence_length,vocab)
    # 选择优化器
    optim = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
    log = []
    # # 训练过程
    for epoch in range(epoch_num):
    # for epoch in range(1):
        starttime = time.time()
        my_model.train()
        my_loss = []
        for batch_num in range(int(train_sample/batch_size)):
            # print(f"batch{batch_num}开始")
            dataset_x, dataset_y = build_dataset(batch_size, vocab, sentence_length) #构造样本数据集
            optim.zero_grad()                                        # 梯度归零
            batch_loss = my_model(dataset_x,dataset_y)               # 前向传播
            # print(batch_loss.item())
            batch_loss.backward()                                    # 计算梯度
            optim.step()                                             # 更新权重
            my_loss.append(batch_loss.item())
        # 输出本轮测试的成绩
        endtime = time.time()
        print("="*20+"\n"+"第%d轮训练平均loss:%f,用时:%fs"%(epoch+1,np.mean(my_loss),endtime-starttime))
        rat = evaluate(my_model, vocab, sentence_length)
        log.append([rat,np.mean(my_loss)])
    # #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(my_model.state_dict(), "model_3.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()


#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = TorchModel(char_dim,sentence_length,vocab)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab["unk"]) for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%s" % (input_string, int(torch.argmax(result[i])), result[i])) #打印结果



if __name__ == '__main__':
    main()
    input_string = ['acdfgh','sjkfhd','cdkgtf','xinfrd']
    predict('model_3.pth','vocab.json',input_string)