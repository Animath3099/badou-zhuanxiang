# -*- coding: utf-8 -*-

import torch
import csv
import random
import os
import time
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data, load_data_k_fold, csv2json
from sklearn.model_selection import KFold

#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


'''
在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。如果设置初始化，则每次初始化都是固定的。
实际上，计算机并不能产生真正的随机数，而是已经编写好的一些无规则排列的数字存储在电脑里，把这些数字划分为若干相等的N份，并为每份加上一个编号，编号固定的时候，获得的随机数也是固定的。
使用原因：在需要生成随机数据的实验中，每次实验都需要生成数据。设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。
————————————————
版权声明：本文为CSDN博主「初识-CV」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_38410428/article/details/119569425
'''
seed = Config["seed"]
random.seed(seed)                   # python的内置模块，生成随机数
np.random.seed(seed)                # 在Numpy内部也有随机种子，当你使用numpy中的随机数的时候，可以通过此方式固定
# 需要注意的是当只调用torch.cuda.manual_seed()一次时并不能生成相同的随机数序列。如果想要得到相同的随机数序列就需要每次产生随机数的时候都要调用一下torch.cuda.manual_seed()。
# torch.mamual_seed(seed)             # 为CPU中设置种子，生成随机数     # 当设置的种子固定下来的时候，之后依次pytorch生成的随机数序列也被固定下来。
# torch.cuda.manual_seed(seed)        # 为特定GPU设置种子，生成随机数
torch.cuda.manual_seed_all(seed)    # 为所有GPU设置种子，生成随机数


def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    # train_data = load_data(config["train_data_path"], config)
    # train_data = load_data('../test.json', config)

    logger.info("当前k折交叉验证k值：%d" % (config["k_fold_number"]))
    train_data, test_data = load_data_k_fold(config["data_path"], config, config["k_fold_number"])
    logger.info("总数据量：%d, 训练集数据量：%d, 测试集数据量：%d" % (len(train_data) + len(test_data), len(train_data), len(test_data)))

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
    evaluator = Evaluator(config, model, logger, test_data)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:   # 抽查 loss
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

def dict2csv(my_dict, csvfile):
    with open(csvfile, 'a+', encoding='utf8') as f:
        [f.write('{0},{1}\n'.format(key, value)) for key, value in my_dict.items()]

def write_csv(csvfile, line):
    with open(csvfile, 'a+', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(line)

if __name__ == "__main__":
    # main(Config)

    # # for model in ["gated_cnn", "cnn"]:
    # for model in ["RCNN", "StackGatedCNN", "BertLSTM", "BertMidLayer"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    # csv转json
    csvfile = "../文本分类练习.csv"
    jsonfile = '../test.json'
    Config["data_path"] = csv2json(csvfile, jsonfile)

    #对比所有模型
    # 写表头
    title = ["model", "最后一轮准确率", 'model_path', 'data_path', 'vocab_path', 'model_type', 'max_length', 'hidden_size',
             'kernel_size', 'num_layers', 'epoch', 'batch_size', 'pooling_style', 'optimizer', 'learning_rate',
             'pretrain_model_path', 'seed', 'k_fold_number', 'class_num', 'vocab_size', '耗时']
    write_csv('model_comparison.csv', title)

    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    for model in ["gated_cnn", "cnn", "fast_text", "lstm", "gru", "rnn", "stack_gated_cnn", "rcnn", "bert", "bert_lstm",
                  "bert_cnn", "bert_mid_layer"]:
        Config["model_type"] = model
        start_time = time.time()
        for lr in [1e-3]:
            Config["learning_rate"] = lr
            # for hidden_size in [128, 256]:
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg"]:
                        Config["pooling_style"] = pooling_style
                        for epoch in [15]:
                            Config["epoch"] = epoch
                            for optimizer in ["adam"]:
                                Config["optimizer"] = optimizer
                                for k_fold_number in [7,]:
                                    Config["k_fold_number"] = k_fold_number
                                    result = main(Config)
                                    end_time = time.time()
                                    line = [model, str(result)] + [str(v) for k,v in Config.items()] + [end_time - start_time]
                                    write_csv('model_comparison.csv', line)
                                    print(model, "最后一轮准确率：", main(Config), "当前配置：", Config)


