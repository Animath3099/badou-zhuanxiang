import numpy as np
import random
import sys
import jieba
import math
import copy
'''
Kmeans算法实现
'''
class KMeansClusterer:  # k均值聚类
    def __init__(self, elements, cluster_num):
        self.ndarray = self.__preprocess_elements(elements)
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(self.ndarray, cluster_num)
        self.distance = self.__compute_distance(self.ndarray)
    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for idx,item in enumerate(self.ndarray):
            distance_min = sys.maxsize
            index = -1
            for center_idx,i in enumerate( self.points):
                distance = self.distance[idx][i]
                if distance < distance_min:
                    distance_min = distance
                    index = center_idx
            result[index] = result[index] + [idx]
        new_center = []
        for item in result:
            new_center.append(self.__center(item))
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center):
            sum_distance = self.__sumdis(result)
            return result, self.points, sum_distance
        self.points = copy.deepcopy(new_center)
        return self.cluster()

    def __sumdis(self,result):
        #计算总距离和
        sum_distance=0
        for idx,i in enumerate(self.points):
            for j in result[idx]:
                sum_distance += self.distance[i][j]
        return sum_distance

    def __center(self, elems_idx):
        # 计算簇中的中心点(这个点到其它点的距离和的值最小)
        sum_distance = []
        for i in elems_idx:
            tmp_sum_distance = 0.0
            for j in elems_idx:
                tmp_sum_distance += self.distance[i][j]
            sum_distance.append(tmp_sum_distance)
        sum_distance = np.array(sum_distance)
        min_idx = np.argmin(sum_distance)
        return elems_idx[min_idx]

    def __distance(self, list_of_words1, list_of_words2):
        #计算两个句子距离，使用jaccard距离
        return 1-len(set(list_of_words1) & set(list_of_words2))/len(set(list_of_words1) | set(list_of_words2))

    def __preprocess_elements(self,elements):
        narray = []
        for element in elements:
            element = element.strip().split()
            narray.append(element)
        return narray

    def __compute_distance(self,narray):
        cnt = len(narray)
        distance = np.zeros((cnt,cnt))
        for i in range(cnt):
            for j in range(cnt):
                if j >= i:
                    if i == j:
                        continue
                    else:
                        distance[i][j] = self.__distance(narray[i],narray[j])
                else:
                    distance[i][j] = distance[j][i]
        return distance

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > len(ndarray):
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0, len(ndarray), step=1).tolist(), cluster_num)
        '''
        points = []
        for index in indexes:
            points.append(ndarray[index])
        '''
        return indexes

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences
#加载语料
sentences = load_sentence("./titles.txt")
print(sentences)
n_clusters = int(math.sqrt(len(sentences)))   #聚类数量
print("n_clusters=%d" % n_clusters)
kmeans = KMeansClusterer(sentences, n_clusters)
result, centers, distances = kmeans.cluster()
print(result)
print(centers)
print(distances)
#打印每个簇中包含的标题
sentences = list(sentences)
for idx,item in enumerate(result):
    print("====cluster:%d===========" % idx)
    for i in item:
        if i == centers[idx]:
            print("center:%s" % sentences[i])
        else:
            print("%s" % sentences[i])
