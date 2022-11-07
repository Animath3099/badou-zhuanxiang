"""
configuration information
"""

Config={
    "model_path":"output/",
    "train_data_path":"text_classification/train.csv",
    "valid_data_path":"text_classification/test.csv",
    "predict_data_path":"text_classification/verification.csv",
    "vocab_path":"vocab.txt",
    "vocab_size":21128,
    "class_num":2,
    "model_type":"gru",
    "max_length":30,
    "hidden_size":128,
    "kernel_size":3,
    "num_layers":2,
    "epoch":10,
    "batch_size":64,
    "pooling_style":"max",
    "optimizer":"adam",
    "learning_rate":1e-5,
    "pretrain_model_path":r"./bert-base-chinese/",
    "seed":987
}