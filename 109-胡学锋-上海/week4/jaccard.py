import numpy as np
import random
import sys
'''
Kmeans算法实现
原文链接：https://blog.csdn.net/qingchedeyongqi/article/details/116806277
'''

class KMeansClusterer:  # k均值聚类
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray:
            distance_min = 1 #jaccard 最大距离1
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item]
        new_center = []
        for item in result:
            new_center.append(self.__center(item))
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center):
            sum = self.__sumdis(result)
            return result, self.points, sum
        self.points = (new_center)
        return self.cluster()

    def __sumdis(self,result):
        #计算总距离和
        sum=0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum+=self.__distance(result[i][j],self.points[i])
        return sum

    # def __center(self, list):
    #     # 计算每一列的平均值
    #     return np.array(list).mean(axis=0)
    def __center(self, cluster_sentences):
        center = "             "    #设置一个不存在的占位字符
        distance_to_all = 999999999  #占位最大距离
        for sentence_a in cluster_sentences:
            distance = 0
            for sentence_b in cluster_sentences:
                distance += self.__distance(sentence_a, sentence_b)
            distance /= len(cluster_sentences)
            if distance < distance_to_all:
                center = sentence_a
                distance_to_all = distance
        return center

    # def __distance(self, p1, p2):
    #     #计算两点间距
    #     tmp = 0
    #     for i in range(len(p1)):
    #         tmp += pow(p1[i] - p2[i], 2)
    #     return pow(tmp, 0.5)

    def __distance(self,list_of_words1, list_of_words2):
        # 计算两点间距jaccard
        return 1 - len(set(list_of_words1) & set(list_of_words2)) / len(set(list_of_words1) | set(list_of_words2))
    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > len(ndarray):
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0, len(ndarray), step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index])
        return points

# x = np.random.rand(100, 8)
# kmeans = KMeansClusterer(x, 10)
# result, centers, distances = kmeans.cluster()
# print(result)
# print(centers)
# print(distances)

if __name__ == '__main__':
    import jieba
    f = open("titles.txt",encoding="utf-8")
    lines = f.readlines()
    words = []
    for i in lines:
        words.append(jieba.lcut(i))
    # print(words)
    k = KMeansClusterer(words,100)
    result, points, sum = k.cluster()
    print(result[5])
    print(points)
    # print(sum)
    f.close()