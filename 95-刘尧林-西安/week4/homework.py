#类内距离计算

import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

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


def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算


    #实现类内距离计算
    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)
    for vector, sentence, label in zip(vectors, sentences, kmeans.labels_):  #取出向量、句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
        vector_label_dict[label].append(vector)

    distance_dict = {}  #存放每个类的类内距离
    for label, vectors in vector_label_dict.items(): #计算类内距离
        sentences_arr = np.array(vectors)
        distance = np.power(sentences_arr-sentences_arr.mean(0),2)
        mean_distance = np.sqrt(distance.sum(1)).mean()
        distance_dict[label] = mean_distance

    distance_tuplelist_sorted = sorted(distance_dict.items(),key=lambda x: x[1], reverse=False) #对类内距离排序

    cluster_num = 10 #取出类的个数
    for i in range(cluster_num):
        label = distance_tuplelist_sorted[i][0]
        sentences = sentence_label_dict[label]
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()