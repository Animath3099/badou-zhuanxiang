#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2022/9/3 16:12
@Author  : luoshixiang
@Email   : just_rick@163.com
@File    : week04_intra-class_distance_sorting.py
@effect  : 在kmeans聚类基础上，实现根据类内距离排序，输出结果
"""

import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
import scipy.spatial.distance as dist

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


# 欧式距离,可理解成点到点的直线距离
def educlidean_distance(x, y):
    # return np.sqrt(np.sum(np.square(x - y)))
    return np.sqrt(np.sum((x - y)*((x - y).T)))

# 标准欧式距离
def educlidean_distance2(x, y):
    return np.sqrt(np.sum((x - y) ** 2))
    # return np.linalg.norm(x - y)

# 余弦距离
def cosine_distance(x, y):
    # 向量的余弦相似度（夹角的余弦值）
    cos_similiarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return 1 - cos_similiarity  # 余弦距离

# def angle(vec1, vec2, degree=False):
#     '''向量夹角，通过degree=True切换弧度制到角度制'''
#     angle = np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*(np.linalg.norm(vec2))))
#     if degree:
#         angle = angle*180/np.pi
#     return angle

# 切比雪夫距离
def chebyshev_distance(x, y):
    return np.max(np.abs(x - y))
    # return np.linalg.norm(x-y,ord=np.inf)

# 曼哈顿距离,也称为城市街区距离， 指的是两点之间的实际距离（不一定是直线）
def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))
    # return np.linalg.norm(x-y, ord=1)

def hamming_distance(x,y):  # 此案例中不适用
    '''两个等长字符串s1与s2之间的汉明距离定义为将其中一个变为另外一个所需要作的最小替换次数。例如字符串“1111”与“1001”之间的汉明距离为2。
    应用：信息编码（为了增强容错性，应使得编码间的最小汉明距离尽可能大）。'''
    return np.shape(np.nonzero(x - y)[0])[0]

def jaccard_distance(x, y): # 此案例中不适用
    '''杰卡德距离用两个集合中不同元素占所有元素的比例来衡量两个集合的区分度。'''
    return dist.pdist(np.array([x, y]), 'jaccard')

def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    index = 0
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_vec = vectors[index]   # 每个句子的向量
        cluster_centers = kmeans.cluster_centers_[label]  # 聚类中心点向量
        # 计算每句话到中心的距离
        # distance_to_center = educlidean_distance(sentence_vec, cluster_centers)  # 欧式距离
        # distance_to_center = educlidean_distance2(sentence_vec, cluster_centers)  # 标准欧式距离
        # distance_to_center = cosine_distance(sentence_vec, cluster_centers)  # 余弦距离
        distance_to_center = chebyshev_distance(sentence_vec, cluster_centers)  # 切比雪夫距离
        # distance_to_center = manhattan_distance(sentence_vec, cluster_centers)  # 曼哈顿距离
        # distance_to_center = chebyshev_distance(sentence_vec, cluster_centers)  # 汉明距离，此案例中不适用
        # distance_to_center = manhattan_distance(sentence_vec, cluster_centers)  # 杰卡德距离，此案例中不适用
        sentence_label_dict[label].append([sentence, distance_to_center])         #同标签的放到一起
        index += 1

    # 每个聚类内句子数量不一致，为了可比较，取平均值
    avg_distance = defaultdict(list)
    for label, sentence_and_distance in sentence_label_dict.items():
        distance_sum = sum([sd[1] for sd in sentence_and_distance])
        sentence_num = len(sentence_and_distance)
        avg_distance[label].append([sentence_and_distance, distance_sum/sentence_num])
    # 类内平均距离按升序排列
    avg_distance_sorted = sorted(avg_distance.items(), key=lambda x: x[1][0][1])
    for label, sentences in avg_distance_sorted:
        print(f"cluster {label}, 类内平均距离：{sentences[0][1]}")
        select_sentences = sentences[0][0]
        for i in range(min(10, len(select_sentences))):  # 随便打印几个，太多了看不过来
            print(select_sentences[i][0].replace(" ", ""), select_sentences[i][1])
        print("---------")


if __name__ == "__main__":
    main()

