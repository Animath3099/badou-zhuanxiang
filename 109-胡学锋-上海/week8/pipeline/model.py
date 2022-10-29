import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel
from torchcrf import CRF
"""
建立网络模型结构
"""
class TorchModel(nn.Module):
    def __init__(self,config):
        super(TorchModel,self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        self.class_num = config["class_num"]
        self.model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        self.use_crf = config["use_crf"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if self.model_type == "fast_text":
            self.encoder = lambda x: x
        elif self.model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        elif self.model_type == "bi-lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,bidirectional=True)
        elif self.model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        elif self.model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers)
        elif self.model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"],return_dict=False)
            hidden_size = self.encoder.config.hidden_size

        if str(self.model_type).startswith("bi-"):
            self.classify = nn.Linear(hidden_size*2, self.class_num)
        else:
            self.classify = nn.Linear(hidden_size, self.class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy  #loss采用交叉熵损失
        self.crf_layer = CRF(self.class_num, batch_first=True)
        self.bi_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, bidirectional=False)
        # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        if self.use_bert:  # bert返回的结果是 (sequence_output, pooler_output)
            x = self.encoder(x)[0]
            x = self.bi_lstm(x)
        else:
            x = self.embedding(x)  # input shape:(batch_size, sen_len)
            x = self.encoder(x)  # input shape:(batch_size, sen_len, input_dim)

        if isinstance(x, tuple):  # RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]
        # 可以采用pooling的方式得到句向量
        # if self.pooling_style == "max":
        #     self.pooling_layer = nn.MaxPool1d(x.shape[1])
        # else:
        #     self.pooling_layer = nn.AvgPool1d(x.shape[1])
        # x = self.pooling_layer(x.transpose(1, 2)).squeeze()  # input shape:(batch_size, sen_len, hidden_size)

        # 也可以直接使用序列最后一个位置的向量
        # x = x[:, -1, :]

        predict = self.classify(x)  # input shape:(batch_size,sen_len, input_dim)
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict

#优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)