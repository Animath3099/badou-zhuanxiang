import json

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import Ethan,choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def train(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = Ethan(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return model, train_data
def predict(config):
    while True:
        usr_sentence=input("请输入您的句子:")
        sentence_length=len(usr_sentence)
        usr_txt=open("usr_sentence.txt",mode="w")
        usr_txt.write(usr_sentence)
        usr_txt.close()
        input_sentence=load_data("usr_sentence.txt",config,shuffle=True)
        #加载训练好的模型
        model_name=Config["model_path"]+"/epoch_2000.pth"
        model=Ethan(config)
        model.load_state_dict(torch.load(model_name))
        model.eval()
        result=[]
        #读入schema
        with open(config["schema_path"],encoding="utf8") as f:
            schema=json.load(f)
        schema_inverse=dict((value,key) for key ,value in schema.items())
        with torch.no_grad():
            predict_result=model.forward(torch.LongTensor(input_sentence.dataset.data))
            # predict_result=torch.argmax(predict_result)
            predict_result=predict_result.view(-1,predict_result.shape[-1])
            predict_result=predict_result.tolist()
            result=[]
            for i in range(sentence_length):
                predict = predict_result[i]
                max_value = max(predict)
                maxId = predict.index(max_value)
                word=usr_sentence[i]+" " +schema_inverse[maxId]+" /"
                result.append(word)
            print("您输入的句子的\"命名实体识别\"的结果为:",result)


if __name__ == "__main__":
    # model, train_data = train(Config)
    predict(Config)
