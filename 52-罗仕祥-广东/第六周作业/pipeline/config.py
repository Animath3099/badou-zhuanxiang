# -*- coding: utf-8 -*-

"""
配置参数信息
E:\02 AI\badou-zhuanxiang-main\52-罗仕祥-广东\week6\chinese-bert-wwm-ext
"""

Config = {
    "model_path": "output",
    "data_path": "../test.json",
    "vocab_path":"../chars.txt",
    "model_type":"gated_cnn",
    "max_length": 20,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 64,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"E:\02 AI\badou-zhuanxiang-main\52-罗仕祥-广东\第六周作业\bert-base-chinese",
    "seed": 987,
    "k_fold_number": 7
}