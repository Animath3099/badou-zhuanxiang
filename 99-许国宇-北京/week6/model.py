import torch
import torch.nn as nn
from torch.optim import Adam,SGD
from transformers import BertModel
from config import Config
from evaluate import Evaluator
from loader import load_data
import random
import numpy as np
import logging
import os

"""
make the network model
"""
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)

"""
Model training main program
"""
seed=Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



class TorchModel(nn.Module):
    def __init__(self,Config):
        nn.Module.__init__(self)
        hidden_size=Config["hidden_size"]
        vocab_size=Config["vocab_size"]+1
        class_num=Config["class_num"]
        model_type=Config["model_type"]
        num_layers=Config["num_layers"]
        self.use_bert=False;
        self.embedding=nn.Embedding(vocab_size,hidden_size,padding_idx=0)
        if model_type=="fast_text":
            self.encoder=lambda x:x
        elif model_type=="lstm":
            self.encoder=nn.LSTM(hidden_size,hidden_size,num_layers=num_layers)
        elif model_type=="gru":
            self.encoder=nn.GRU(hidden_size,hidden_size,num_layers=num_layers)
        elif model_type=="rnn":
            self.encoder=nn.RNN(hidden_size,hidden_size,num_layers=num_layers)
        elif model_type=="cnn":
            self.encoder=CNN(Config)
        elif model_type=="gated_cnn":
            self.encoder=GatedCNN(Config)
        elif model_type=="stack_gated_cnn":
            self.encoder=StackGatedCNN(Config)
        elif model_type=="rcnn":
            self.encoder=RCNN(Config)
        elif model_type=="bert":
            self.use_bert=True
            self.encoder=BertModel.from_pretrained(Config["pretrain_model_path"])
            hidden_size=self.encoder.config.hidden_size
        elif model_type=="bert_lstm":
            self.use_bert=True
            self.encoder=BertLSTM(Config)
            hidden_size=self.encoder.bert.config.hidden_size
        elif model_type=="bert_cnn":
            self.use_bert=True
            self.encoder=BertCNN(Config)
            hidden_size=self.encoder.bert.config.hidden_size
        elif model_type=="bert_mid_layer":
            self.use_bert=True
            self.encoder=BertMidLayer(Config)
            hidden_size=self.encoder.bert.config.hidden_size

        self.classify=nn.Linear(hidden_size,class_num)
        self.pooling_style=Config["pooling_style"]
        self.loss=nn.functional.cross_entropy

    #when entering the real label,return loss value,no real label,return the predict value
    def forward(self,x,target=None):
        if self.use_bert: #the result that bert returned is(sequence_output,pooler_output)
            x=self.encoder(x)
            if Config["model_type"]=="bert":
                x=x["last_hidden_state"]
        else:
            x=self.embedding(x) #input shape(batch_size,sen_len)
            x=self.encoder(x) #input shape:(batch_size,sen_len,input_dim)

        if isinstance(x,tuple):#the model of RNN class will also return the hidden unit vector, we only take the sequence result
            x=x[0]
        #sentences vectors can be obtained by pooling
        if self.pooling_style=="max":
            self.pooling_layer=nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer=nn.AvgPool1d(x.shape[1])
        x=self.pooling_layer(x.transpose(1,2)).squeeze() #input shape(batch_size,sen_len,input_dim)

        #you can also just use the vector at the last position in the sequence
        #x=x[:,-1,:]
        predict=self.classify(x) #input shape(batch_size,input_dim)
        if target is not None:
            return self.loss(predict,target.squeeze())
        else:
            return predict

    def myTrain(self):
        model_type = ["fast_text", "lstm", "gru", "rnn", "cnn", "gated_cnn", "stack_gated_cnn", "rcnn", "bert",
                      "bert_lstm", "bert_cnn", "bert_mid_layer"]
        for model_t in model_type:
            model_name="output/"+model_t+".pth"
            Config["model_type"]=model_t
            print(model_name + "模型开始训练:------------------------------")
            # Create a directory to hold the model
            if not os.path.isdir(Config["model_path"]):
                os.mkdir(Config["model_path"])
            # loading training data
            # if config["model_type"] !="bert":
            train_data = load_data(Config["train_data_path"], Config)
            # load the model
            model = TorchModel(Config)
            # Indicates whether the GPU is used
            cuda_flag = torch.cuda.is_available()
            if cuda_flag:
                logger.info("GPU available,migrate model to GPU")
                model = model.cuda()
            # loading the optimizer
            optimizer = choose_optimizer(Config, model)
            # loading the effects test data
            evaluator = Evaluator(Config, model, logger)
            # training
            for epoch in range(Config["epoch"]):
                epoch += 1
                model.train()
                logger.info("epoch %d begin" % epoch)
                train_loss = []
                for index, batch_data in enumerate(train_data):
                    if cuda_flag:
                        batch_data = [d.cuda() for d in batch_data]
                    optimizer.zero_grad()
                    input_ids, labels = batch_data
                    loss = model(input_ids, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item())
                    if index % int(len(train_data) / 2) == 0:
                        logger.info("batch loss %f" % loss)
                logger.info("epoch average loss:%f" % np.mean(train_loss))
                acc = evaluator.eval(epoch)
            torch.save(model.state_dict(), model_name)
        return acc
    def predict(self):
        model_type = ["fast_text", "lstm", "gru", "rnn", "cnn", "gated_cnn", "stack_gated_cnn", "rcnn", "bert",
                      "bert_lstm", "bert_cnn", "bert_mid_layer"]
        for model_t in model_type:
            model_name = Config["model_path"]+model_t + ".pth"
            Config["model_type"]=model_t
            print(model_name + "模型预测结果:")
            model = TorchModel(Config)
            model.load_state_dict(torch.load(model_name))
            model.eval()
            verification_data=load_data(Config["predict_data_path"],Config)
            number=0
            for index,batch_data in enumerate(verification_data):
                review_vec,label=batch_data
                correct_num=0
                with torch.no_grad():
                    predict_label = model.forward(torch.LongTensor(review_vec))
                    predict_result=predict_label.tolist()
                    label=label.tolist()
                    for index,predict_list in enumerate(predict_result):
                        if predict_list[0]>predict_list[1]:
                            if label[index][0]==0:
                                correct_num += 1
                        else:
                            if label[index][0]==1:
                                correct_num += 1
                number+=1
                print("预测正确率为:", correct_num / Config["batch_size"])
                if number==10:#限制预测次数,否则计算时间太长
                    break

class CNN(nn.Module):
    def __init__(self,config):
        nn.Module.__init__(self)
        hidden_size=config["hidden_size"]
        kernel_size=config["kernel_size"]
        pad=int((kernel_size-1)/2)
        self.cnn=nn.Conv1d(hidden_size,hidden_size,kernel_size,bias=False,padding=pad)
    def forward(self,x):#x:(batch_size,max_len,embeding_size)
        return self.cnn(x.transpose(1,2)).transpose(1,2)

class GatedCNN(nn.Module):
    def __init__(self,config):
        nn.Module.__init__(self)
        self.cnn=CNN(config)
        self.gate=CNN(config)

    def forward(self,x):
        a=self.cnn(x)
        b=self.gate(x)
        b=torch.sigmoid(b)
        return torch.mul(a,b)

class StackGatedCNN(nn.Module):
    def __init__(self,config):
        nn.Module.__init__(self)
        self.num_layers=config["num_layers"]
        self.hidden_size=config["hidden_size"]
        #multiple models can be placed within the modelList,which is accessed like a list
        self.gcnn_layers=nn.ModuleList(GatedCNN(config) for i in range(self.num_layers))
        self.ff_linear_layers1=nn.ModuleList(nn.Linear(self.hidden_size,self.hidden_size) for i in range(self.num_layers))
        self.ff_linear_layers2=nn.ModuleList(nn.Linear(self.hidden_size,self.hidden_size) for i in range(self.num_layers))
        self.bn_after_gcnn=nn.ModuleList(nn.LayerNorm(self.hidden_size) for i in range(self.num_layers))
        self.bn_after_ff=nn.ModuleList(nn.LayerNorm(self.hidden_size) for i in range(self.num_layers))

    def forward(self,x):
        #following Bert's Transformer model structure,self-attention is replaced by GCNN
        for i in range(self.num_layers):
            gcnn_x=self.gcnn_layers[i](x)
            x=gcnn_x+x
            x=self.bn_after_gcnn[i](x)
            #like the feed-forward layer,two linear layers are used
            l1=self.ff_linear_layers1[i](x)
            l1=torch.relu(l1)
            l2=self.ff_linear_layers2[i](l1)
            x=self.bn_after_ff[i](x+12)
        return x

class RCNN(nn.Module):
    def __init__(self,config):
        nn.Module.__init__(self)
        hidden_size=config["hidden_size"]
        self.rnn=nn.RNN(hidden_size,hidden_size)
        self.cnn=GatedCNN(config)

    def forward(self,x):
        x,_=self.rnn(x)
        x=self.cnn(x)
        return x

class BertLSTM(nn.Module):
    def __init__(self,config):
        nn.Module.__init__(self)
        self.bert=BertModel.from_pretrained(config["pretrain_model_path"])
        self.rnn=nn.LSTM(self.bert.config.hidden_size,self.bert.config.hidden_size,batch_first=True)

    def forward(self,x):
        x=self.bert(x)[0]
        x,_=self.rnn(x)
        return x

class BertCNN(nn.Module):
    def __init__(self,config):
        nn.Module.__init__(self)
        self.bert=BertModel.from_pretrained(config["pretrain_model_path"])
        config["hidden_size"]=self.bert.config.hidden_size
        self.cnn=CNN(config)

    def forward(self,x):
        x=self.bert(x)[0]
        x=self.cnn(x)
        return x

class BertMidLayer(nn.Module):
    def __init__(self,config):
        nn.Module.__init__(self)
        self.bert=BertModel.from_pretrained(config["pretrain_model_path"])
        self.bert.config.output_hidden_states=True

    def forward(self,x):
        layer_states=self.bert(x)[2]
        layer_states=torch.add(layer_states[-2],layer_states[-1])
        return layer_states

#optimizer selection
def choose_optimizer(config,model):
    optimizer=config["optimizer"]
    learning_rate=config["learning_rate"]
    if optimizer=="adam":
        return Adam(model.parameters(),lr=learning_rate)
    elif optimizer=="sgd":
        return SGD(model.parameters(),lr=learning_rate)

if __name__ == "__main__":
    model = TorchModel(Config)
    # model.myTrain()
    model.predict()























