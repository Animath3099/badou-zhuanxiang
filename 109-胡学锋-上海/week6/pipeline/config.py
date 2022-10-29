# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "./data/train.csv",
    "valid_data_path": "./data/valid.csv",
    "vocab_path":"./data/chars.txt",
    "model_type":"lstm",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 3,
    "epoch": 15,
    "batch_size": 64,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\badou\pretrain_model\chinese-bert_chinese_wwm_pytorch",
    "seed": 987,
    "grid_params" : {"a":1,"b":2}
}