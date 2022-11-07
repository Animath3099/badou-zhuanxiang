# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "./data/train.json",
    "valid_data_path": "./data/test.json",
    "vocab_path":"./data/chars.txt",
    # "model_type":"bi-lstm",
    "model_type":"bert",
    "max_length": 128,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 3,
    "epoch": 500,
    "batch_size": 32,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\ProgramData\bert-base-chinese",
    "seed": 987,
    "grid_params" : {"a":1,"b":2},
    "use_crf":False
}