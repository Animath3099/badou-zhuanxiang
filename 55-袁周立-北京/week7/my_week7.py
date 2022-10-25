import os

import numpy as np
import torch.cuda
import logging
from loader import load_data
from model import SentenceMatch
from evaluate import Evaluate


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Config = {
    "model_path": "model_output",
    "schema_path": "data/schema.json",
    "train_data_path": "data/train.json",
    "valid_data_path": "data/valid.json",
    "vocab_path": "chars.txt",
    "max_length": 15,
    "batch_size": 32,
    "epoch_data_size": 200,
    "hidden_size": 128,
    "epoch": 10,
    "optimizer": "adam",
    "learning_rate": 1e-3
}


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    train_data = load_data(config["train_data_path"], config)
    model = SentenceMatch(config)

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，模型迁移至gpu")
        model = model.cuda()

    optimizer = model.choose_optimizer()
    evaluate = Evaluate(config, model, logger, train_data.dataset.get_knwb())

    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("开始第%d轮训练：" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            optimizer.zero_grad()
            sample1, sample2, labels = batch_data
            loss = model(sample1, sample2, labels)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        logger.info("第%d轮平均损失：%f" % (epoch, np.mean(train_loss)))
        evaluate.eval(epoch)

    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main(Config)
