#!/usr/bin/env python3
# coding: utf-8

# 2.在不进行文本向量化的前提下对文本进行kmeans聚类(可采用jaccard距离)

import sys
import math
import random
import numpy as np

from word2vec_kmeans import load_sentence
from collections import defaultdict


class KmeansClass:
    def __init__(self, sentences, class_num):
        self.sentences = sentences
        self.class_num = class_num
        self.curr_centers = random.sample(sentences, class_num)
        self.curr_classes = defaultdict(list)
        self.curr_classes = {i: [sent] for i, sent in enumerate(self.curr_centers)}

    def jaccard_distance(self, list_of_words1, list_of_words2):
        return 1 - len(set(list_of_words1) & set(list_of_words2)) / len(set(list_of_words1) | set(list_of_words2))

    def find_new_center(self):
        new_center = list()
        for label, curr_sents in self.curr_classes.items():
            curr_cent = curr_sents[0]
            curr_dist = 999999999
            for sent_1 in curr_sents:
                avg_dist = np.sum([self.jaccard_distance(sent_1, sent_2) for sent_2 in curr_sents]) / len(curr_sents)
                if avg_dist < curr_dist:
                    curr_cent = sent_1
                    curr_dist = avg_dist

            new_center.append(curr_cent)
        return new_center

    def fit(self):
        count = 0
        while True:
            count += 1
            self.curr_classes = defaultdict(list)
            # 计算到各类的距离，并归类
            for sent in self.sentences:
                all_dis_list = [self.jaccard_distance(sent, cen) for cen in self.curr_centers]
                min_dis_idx = np.argmin(all_dis_list)
                self.curr_classes[min_dis_idx].append(sent)

            # 迭代到新聚类中心没有更新为止
            new_center = self.find_new_center()

            flags = [False if c not in new_center else True for c in self.curr_centers]
            print('training ... , fit:', count, 'flags:', + flags.count(True))
            if flags.count(True) == self.class_num:
                break

            self.curr_centers = new_center


def main():
    sentences = [sent.split() for sent in load_sentence('titles.txt')]
    class_num = 10
    kmeans = KmeansClass(sentences, class_num)

    kmeans.fit()

    for lab, curr_class_list in kmeans.curr_classes.items():
        print('\n聚类标签: ', lab, ', 句子数量: ', len(curr_class_list))
        print('\n'.join([''.join(sent) for sent in random.sample(curr_class_list, min(len(curr_class_list), 5))]))


if __name__ == "__main__":
    main()
