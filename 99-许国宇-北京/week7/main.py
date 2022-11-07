import json

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data,load_schema

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def train(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id, labels)
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
            loss.backward()
            # print(loss.item())
            # print(model.classify.weight.grad)
            optimizer.step()
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "representation_epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return model, train_data
def predict(Config):
    while True:
        user_question=input("请输入您的问题:")
        usr_txt=open("usr_question.txt",mode="w")
        usr_txt.write(user_question)
        usr_txt.close()
        input_id = load_data("usr_question.txt",Config,shuffle=True)
        #加载标准问集合
        model_name = Config["model_path"] + "representation_epoch_100.pth"
        model = TorchModel(Config)
        model.load_state_dict(torch.load(model_name))
        model.eval()
        schema=load_schema(Config["schema_path"])
        key_value_change_schema=dict((y,x) for x,y in schema.items())
        with torch.no_grad():
            predict = model.forward(torch.LongTensor(input_id.dataset.data))
            predict=torch.argmax(predict)
            predict=key_value_change_schema[predict.tolist()]
            print("您问的问题的标准问为:",predict)

if __name__ == "__main__":
    # model, train_data = train(Config)
    predict(Config)