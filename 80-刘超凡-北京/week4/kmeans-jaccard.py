import numpy
import numpy as np
import random
import sys
import re

'''
Kmeans算法实现
原文链接：https://blog.csdn.net/qingchedeyongqi/article/details/116806277
'''


class KMeansClusterer:  # k均值聚类
    def __init__(self, sentences, cluster_num):
        self.sentences = sentences
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(sentences, cluster_num)
        self.buffer = {}

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.sentences:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index].append(item)
            # print(result[index])
        new_center = []
        for item in result:
            new_center.append(self.__center(item))
        # 中心点未改变，说明达到稳态，结束递归
        print('self.points', self.points)
        print('new_center', new_center)
        if self.points == new_center:
            print('稳定了')
            # sum = self.__sumdis(result)
            return result, self.points
        self.points = new_center
        return self.cluster()

    def __sumdis(self, result):
        # 计算总距离和
        sum = 0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum += self.__distance(result[i][j], self.points[i])
        return sum

    def __all_distance(self, target, list):
        sum = 0
        for item in list:
            sum += self.__distance(target, item)
        return sum

    def __center(self, list):
        # 计算每一列的平均值
        # print('传入的List', len(list))
        if len(list) == 0:
            return "             "
        sum_list = []
        for item in list:
            sum_list.append(self.__all_distance(item, list) / len(sentences))

        target_index = -1;
        min = max(sum_list)
        for index in range(len(sum_list)):
            distance = sum_list[index]
            if distance < min:
                min = distance
                target_index = index

        # print('中心是', list[target_index])
        return list[target_index]

    def __distance(self, p1, p2):
        # 计算两点间距
        # tmp = 0
        # for i in range(len(p1)):
        #     tmp += pow(p1[i] - p2[i], 2)
        # return pow(tmp, 0.5)
        if p1 + p2 in self.buffer:
            return self.buffer[p1 + p2]
        elif p2 + p1 in self.buffer:
            return self.buffer[p2 + p1]
        else:
            # distance = jaccard(p1, p2)
            # 使用了错误的jaccard一直无法稳定，改用参考答案中的实现
            distance = 1 - len(set(p1) & set(p2)) / len(set(p1).union(set(p2)))
            self.buffer[p1 + p2] = distance
            return distance

    def __pick_start_point(self, sentences, cluster_num):
        if cluster_num < 0 or cluster_num > len(sentences):
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(range(0, len(sentences)), cluster_num)
        points = []
        for index in indexes:
            points.append(sentences[index])
        return points


x = np.random.rand(100, 8)


# kmeans = KMeansClusterer(x, 10)
# result, centers, distances = kmeans.cluster()
# print(result)
# print(centers)
# print(distances)

def load_sentence(path):
    sentences = []
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.append(sentence)
    print("获取句子数量：", len(sentences))
    return sentences

# 错误的jaccard实现
# def jaccard(list1, list2):
#     intersection = len(list(set(list1).intersection(set(list2))))
#     union = (len(list1) + len(list2)) - intersection
#     return float(intersection) / union


if __name__ == "__main__":
    sentences = load_sentence("titles.txt")  # 加载所有标题
    kmeans = KMeansClusterer(sentences, 10)
    kmeans.cluster()
