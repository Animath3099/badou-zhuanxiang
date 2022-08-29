#### 样本文件修改

将之前的正负样本修改为10分类问题，样本及分类如下：

|   x   |  y   |   x    |  y   |
| :---: | :--: | :----: | :--: |
| 'abc' |  0   | 'pqr'  |  5   |
| 'def' |  1   | 'stu'  |  6   |
| 'ghi' |  2   | 'vwx'  |  7   |
| 'jkl' |  3   |  'yz'  |  8   |
| 'mno' |  4   | others |  9   |

对y进行one-hot编码，例如：

y=0，label=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

y=1，label=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

自此，生成x，y样本文件

```python
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 指定哪些字出现时为正样本
    label = []
    if set("abc") & set(x):
        y = 0
    elif set("def") & set(x):
        y = 1
    elif set("ghi") & set(x):
        y = 2
    elif set("jkl") & set(x):
        y = 3
    elif set("mno") & set(x):
        y = 4
    elif set("pqr") & set(x):
        y = 5
    elif set("stu") & set(x):
        y = 6
    elif set("vwx") & set(x):
        y = 7
    elif set("yz") & set(x):
        y = 8
    # 指定字都未出现，则为负样本
    else:
        y = 9
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    for i in range(10):
        label.append(1) if i == y else label.append(0)
    return x, label
```

#### 模型修改

将分类器使用多层线性层，这里线性层数可以根据~~个人喜好~~结果修改。激活函数由sigmoid修改为Softmax。

```python
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # embedding层
        self.pool = nn.AvgPool1d(sentence_length)  # 池化层
        # self.classify = nn.Linear(vector_dim, 1)     #线性层
        # self.classify = nn.Linear(vector_dim, 10)     #线性层
        self.classify = nn.Sequential(
            nn.Linear(vector_dim, 37),
            nn.Linear(37, 64),
            nn.Linear(64, 37),
            nn.Linear(37, 10)
        )
        # self.activation = torch.sigmoid     #sigmoid归一化函数
        self.activation = nn.Softmax()
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值

    def forward(self, x, y=None):
        # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = self.embedding(x)
        # x = self.pool(x.transpose(1, 2)).squeeze() #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim)
        x = self.pool(x).squeeze()
        x = self.classify(x)  # (batch_size, vector_dim) -> (batch_size, 1)
        y_pred = self.activation(x)  # (batch_size, 10) -> (batch_size, 10)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果
```

#### 测试部分修改

由于改为多分类任务，模型判别模式不再是正负样本，这里为了~~偷懒~~方便，不打印生成样本分类，只显示正确数目。

softmax输出长度和类别数目相同，输出数值代表每一个分类结果的置信度，因此选取最大置信度坐标作为分类结果，判断是否分类正确。

```python
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    print("本次预测集中共有200个样本")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            # if float(y_p) < 0.5 and int(y_t) == 0:
            #     correct += 1   #负样本判断正确
            # elif float(y_p) >= 0.5 and int(y_t) == 1:
            #     correct += 1   #正样本判断正确
            if torch.argmax(y_p) == torch.argmax(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct/(correct+wrong)))
    return correct/(correct+wrong)
```

#### 调整参数

对于较简单的任务，使用复杂的模型可能造成过拟合，从而使模型性能下降（可能就是没有到达100准确率的原因，这里就不进行调参）

后续感兴趣的可以进行调参，调整学习率、模型结构。

现在：作业完成。