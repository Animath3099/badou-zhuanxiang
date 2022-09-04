#week4作业

'''
1.在kmeans聚类基础上，实现根据类内距离排序，输出结果
2.在不进行文本向量化的前提下对文本进行kmeans聚类(可采用jaccard距离)
'''

import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


#计算公共词
#前面1-的原因是使得文本相似是距离接近0，文本不同时文本接近1，更接近距离的概念
def jaccard_distance(list_of_words1, list_of_words2):
    return 1 - len(set(list_of_words1) & set(list_of_words2)) / len(set(list_of_words1) | set(list_of_words2))

a = jieba.lcut("今天真倒霉")
b = jieba.lcut("今天太走运了")
# print(jaccard_distance(a, b))


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

def get_center(sentences, model):
    vectors = sentences_to_vectors(sentences, model)
    mean_vector = vectors.mean(axis=0)
    return vectors, mean_vector



def calculate_distance(sentence_label_dict, model):
    distance_dict = {}
    for label, sentences in sentence_label_dict.items():
        distance_dict[label] = 0
        vectors, center = get_center(sentences, model)
        cos_sum = 0
        for vector in vectors:
            cos = vector.dot(center) / np.linalg.norm(vector) * np.linalg.norm(center)
            cos_sum += cos
        distance_dict[label] = cos_sum / len(sentences)
    return sorted(distance_dict.items(), key=lambda x: x[1], reverse=True)


def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    distance_dict = calculate_distance(sentence_label_dict, model)
    for label, cos in distance_dict:
        print(r"cluster: {} cos: {} ".format(label, cos))
        for i in range(min(10, len(sentence_label_dict[label]))):  #随便打印几个，太多了看不过来
            print(sentence_label_dict[label][i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

