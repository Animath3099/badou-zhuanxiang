import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


class TorchModelW2(nn.Module):
    def __init__(self,charVectorDim,charNumber,charSet):
        super(TorchModelW2,self).__init__()
        self.embedding=nn.Embedding(len(charSet),charVectorDim)
        self.pool=nn.AvgPool1d(charNumber)
        self.classify=nn.Linear(charVectorDim,3)
        self.activation=torch.softmax
        self.loss=nn.functional.cross_entropy
        self.charNumber=charNumber
        self.firstKind=0
        self.secondKind=0
        self.thirdKind=0

    def forward(self,sentenceX,resultY=None):
        sentenceX=self.embedding(sentenceX)
        sentenceX=self.pool(sentenceX.transpose(1,2)).squeeze()
        sentenceX=self.classify(sentenceX)
        resultY_Pred=self.activation(sentenceX,1)
        if resultY is not None:
            return self.loss(resultY,resultY_Pred)
        else:
            return resultY_Pred

    def build_charSet(self):
        self.chars="abcdefghijklmnopqrstuvwxyz"
        charSet={}
        for index,char in enumerate(self.chars):
            charSet[char]=index
        charSet['unk']=len(charSet)
        self.charSet=charSet
        return charSet

    def build_sample(self):
        sentenceX=[random.choice(list(self.charSet.keys())) for _ in range(self.charNumber)]
        # print("build_sample-------------------------")
        # print("sentenceX=",sentenceX)
        if set("abc") & set(sentenceX):
            if set("xyz") & set(sentenceX):
                resultY=[0,0,1]
                self.thirdKind+=1
            else:
                resultY=[1,0,0]
                self.firstKind+=1
        elif set("xyz") & set(sentenceX):
            if set("abc") & set(sentenceX):
                resultY=[0,0,1]
                self.thirdKind+=1
            else:
                resultY=[0,1,0]
                self.secondKind+=1
        else:
            resultY=[0,0,1]
            self.thirdKind+=1

        # print("resultY=",resultY)
        sentenceX=[self.charSet.get(word,self.charSet['unk']) for word in sentenceX]
        return sentenceX,resultY

    def build_dataSet(self,sampleNumber):
        dataSetX=[]
        dataSetY=[]
        for i in range(sampleNumber):
            x,y=self.build_sample()
            dataSetX.append(x)
            dataSetY.append(y)
        return torch.LongTensor(dataSetX),torch.FloatTensor(dataSetY)

    def evaluate(self,model):
        model.eval()
        sentenceX,resultY=self.build_dataSet(20)
        print("本次预测集中共有%d个样本，其中第一类%d个样本,第二类%d个样本,第三类%d个样本!"%(20, self.firstKind,self.secondKind,self.thirdKind))
        correct,wrong=0,0
        with torch.no_grad():
            resultY_Pred=model(sentenceX)
            for resultY_p,resultY_t in zip(resultY_Pred,resultY):
                for i in range(3):
                    if int(resultY_t[i])==1:
                        if float(resultY_p[i])>=0.5:
                            if i==0:
                                if float(resultY_p[i+1])<0.5 and float(resultY_p[i+2])<0.5:
                                    correct+=1
                                else:
                                    wrong +=1
                            if i==1:
                                if float(resultY_p[i-1])<0.5 and float(resultY_p[i+1])<0.5:
                                    correct+=1
                                else:
                                    wrong +=1
                            if i==2:
                                if float(resultY_p[i-1])<0.5 and float(resultY_p[i-2])<0.5:
                                    correct+=1
                                else:
                                    wrong +=1
                        else:
                            wrong +=1
        print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
        return correct / (correct + wrong)

    def trainN(self):
        trainingNum=500
        batchSize=20
        trainSample=500
        charVectorDim=20
        charNumber=6
        learningRate=0.005
        charSet=self.build_charSet()
        model=TorchModelW2(charVectorDim,charNumber,self.charSet)
        optim=torch.optim.Adam(model.parameters(),lr=learningRate)
        log=[]

        for train in range(trainingNum):
            model.train()
            watchLoss=[]
            for batch in range(int(trainSample/batchSize)):
                x,y=self.build_dataSet(batchSize)
                optim.zero_grad()
                loss=model(x,y)
                loss.backward()
                optim.step()
                watchLoss.append(loss.item())
            print("=========\n第%d轮平均loss:%f" % (train + 1, np.mean(watchLoss)))
        torch.save(model.state_dict(), "modelw2.pth")
        writer = open("charSet.json", "w", encoding="utf8")
        writer.write(json.dumps(charSet, ensure_ascii=False, indent=2))
        writer.close()
        return

    def predict(self,modelPth,charSetPth,inputStrings):
        charVectorDim=20
        charNumber=6
        charSet=json.load(open(charSetPth,"r",encoding="utf8"))
        model=TorchModelW2(charVectorDim,charNumber,self.charSet)
        model.load_state_dict(torch.load(modelPth))
        inputX=[]
        for inputString in inputStrings:
            inputX.append([charSet[char] for char in inputString])
        model.eval()
        with torch.no_grad():
            resultY_pred=model.forward(torch.LongTensor(inputX))
        for i,inputString in enumerate(inputStrings):
            for k in range(3):
                # print(resultY_pred[j,k])
                if float(resultY_pred[i,k]) >=0.5:
                    if k == 0:
                        print("输入：%s, 预测类别：%d, 概率值：%f" % (inputString, 1, resultY_pred[i,k]))
                    elif k == 1:
                        print("输入：%s, 预测类别：%d, 概率值：%f" % (inputString, 2, resultY_pred[i,k]))
                    else:
                        print("输入：%s, 预测类别：%d, 概率值：%f" % (inputString, 3, resultY_pred[i,k]))


if __name__ == "__main__":
    charset="abcdefghijklmnopqrstuvwxyz"
    model=TorchModelW2(20,6,charset)
    model.trainN()
    test_strings = ["rtsshb", "cwsdfg", "rqwdyg", "nlkwww","xyzkkk","qwerty","dfghjk"]
    model.predict("modelw2.pth","charSet.Json",test_strings)
