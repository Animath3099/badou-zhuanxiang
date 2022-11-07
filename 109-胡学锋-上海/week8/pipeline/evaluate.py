# -*- coding: utf-8 -*-
import torch
from loader import load_data
from collections import defaultdict
import numpy as np
import test
"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {name:defaultdict(int) for name in self.valid_data.dataset.b_name }  #用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()

        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_ids) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results,sentences)
        acc = self.show_stats()
        return acc

    def write_stats(self, labels, pred_results,sentences):
        assert len(labels) == len(pred_results)
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label,sentence in zip(labels, pred_results,sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()

            true_label = true_label.cpu().detach().tolist()
            # for y1,y2 in zip(true_label,pred_label):
            #     if int(y1) == int(y2):
            #         self.stats_dict["correct"] += 1
            #     else:
            #         self.stats_dict["wrong"] += 1
            true_entities = self.decode(sentence, true_label)
            # self.logger.info("真实标签："+str(true_entities))
            # self.logger.info("真实值："+str(true_label))

            pred_entities = self.decode(sentence, pred_label)
            # self.logger.info("预测值："+str(pred_label))
            # self.logger.info("预测标签："+str(pred_entities))
            # print(str(true_entities))
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in self.valid_data.dataset.b_name:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def show_stats(self):
        F1_scores = []
        for key in  self.valid_data.dataset.b_name:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in  self.valid_data.dataset.b_name])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in  self.valid_data.dataset.b_name])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in  self.valid_data.dataset.b_name])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return


    def decode(self,sentence,labels):
        # labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        B_idx = [self.valid_data.dataset.label_to_index[i]
                 for i in self.valid_data.dataset.label_to_index
                 if str(i).startswith("B")
                 ]

        idx = [] #实体索引
        left = 0
        while left < len(labels):
            if labels[left] in B_idx:  # 识别实体开头的B
                right = left
                while labels[right + 1] - labels[left] == 1:  # I 紧挨着 B
                    right += 1
                idx.append(labels[left:right + 1])
                b_name = self.valid_data.dataset.index_to_label[labels[left]][2:]
                results[b_name].append(sentence[left:right+1])
                left = right + 1
            else:
                left += 1
        return results