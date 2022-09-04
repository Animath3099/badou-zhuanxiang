#!/usr/bin/env python3
# coding: utf-8

# 1.在kmeans聚类基础上，实现根据类内距离排序，输出结果

import math
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

from word2vec_kmeans import load_word2vec_model, load_sentence, sentences_to_vectors, cosine_distance

# 计算各聚类的类内平均距离，并排序
def avg_dist(kmeans, vectors, sentences):
    dis_dict = defaultdict(list)
    sent_dict = defaultdict(list)
    
    # 计算各聚类内的各个标题到聚类中心的距离
    for i, (lab, sent) in enumerate(zip(kmeans.labels_, sentences)):
        sent_dict[lab].append(sent)
        dis_dict[lab].append(cosine_distance(vectors[i], kmeans.cluster_centers_[lab]))

    avg_dict = defaultdict()
    # 计算平均距离并排序
    for lab, dis_list in dis_dict.items():
        avg_dict[lab] = np.mean(dis_list)
    avg_ord_dict = sorted(avg_dict.items(), key=lambda x: x[1], reverse=True)

    return sent_dict, avg_ord_dict


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sent_dict, avg_order_dict = avg_dist(kmeans, vectors, sentences)

    # 输出结果
    for lab, avg in avg_order_dict:
        print('\n聚类标签: ', lab, ', 类内平均距离: ', avg)
        print('\n'.join([sent.replace(' ', '') for sent in sent_dict[lab][:5]]))


if __name__ == "__main__":
    main()
