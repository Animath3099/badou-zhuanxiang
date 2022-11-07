from config import Config
import logging
import random
import numpy as np
import pandas as pd
import torch
from loader import load_data
from model import TorchModel, choose_optimizer
from evaluate import Evaluator


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config, result, k):
    logger.info("当前k折交叉验证k值：%d" % (k))
    train_data, test_data = load_data(config["data_path"], config, k)
    logger.info("总数据量：%d, 训练集数据量：%d, 测试集数据量：%d" % (len(train_data) + len(test_data), len(train_data), len(test_data)))

    model = TorchModel(config)

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger, test_data)

    final_acc = 0
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
        # torch.save(model.state_dict(), model_path)  #保存模型权重
        print({
            "Model": config["model_type"],
            "Learning_Rate": config["learning_rate"],
            "Hidden_Size": config["hidden_size"],
            "Batch_Size": config["batch_size"],
            "Pooling_Style":  config["pooling_style"],
            "K_Value": k,
            "Acc": acc
        })
        final_acc = acc
    result.append([
        config["model_type"],
        config["learning_rate"],
        config["hidden_size"],
        config["batch_size"],
        config["pooling_style"],
        k,
        final_acc
    ])


if __name__ == "__main__":
    models = ["fast_text", "lstm", "gru", "rnn", "cnn"]
    lrs = [1e-3, 2e-3]
    hidden_sizes = [128, 256]
    batch_sizes = [64]
    pooling_styles = ["avg", "max"]

    result = []
    assert Config["k_min"] < Config["k_max"]
    for model in models:
        Config["model_type"] = model
        for lr in lrs:
            Config["learning_rate"] = lr
            for hidden_size in hidden_sizes:
                Config["hidden_size"] = hidden_size
                for batch_size in batch_sizes:
                    Config["batch_size"] = batch_size
                    for pooling_style in pooling_styles:
                        Config["pooling_style"] = pooling_style
                        print("当前配置：", Config)
                        k = Config["k_min"]
                        while k < Config["k_max"]:
                            main(Config, result, k)
                            k = k + 1
    data_frame = pd.DataFrame(result, columns=["Model", "Learning_Rate", "Hidden_Size", "Batch_Size", "Pooling_Style", "K_Value", "Acc"])
    data_frame.to_csv('result.csv', index=None)
