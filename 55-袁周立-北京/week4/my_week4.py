import numpy as np
import random
import sys

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
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]
        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            sum = self.__sumdis(result)
            k_avg_dis = self.__avg_dis(result)
            return result, self.points, sum, k_avg_dis
        self.points = np.array(new_center)
        return self.cluster()

    def __sumdis(self,result):
        #计算总距离和
        sum=0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum+=self.__distance(result[i][j],self.points[i])
        return sum

    def __avg_dis(self, result):
        # 计算各个类的类内平均距离
        k_avg_dis = []
        for i, item in enumerate(result):
            avg_dis = 0
            for j in item:
                avg_dis += self.__distance(j, self.points[i])
            avg_dis = avg_dis / len(item)
            k_avg_dis.append(avg_dis)
        return k_avg_dis

    def __center(self, list):
        # 取到其他点之和最近的点
        distance_min = 999999999999
        index = -1
        for i in range(len(list)):
            sum_dis = 0
            for j in list:
                sum_dis += self.__distance(list[i], j)
            sum_dis = sum_dis / len(list)
            if sum_dis < distance_min:
                distance_min = sum_dis
                index = i
        return np.array(list[index])

    def __distance(self, p1, p2):
        # 计算两点余弦距离
        p1 = np.array(p1)
        p2 = np.array(p2)
        return 1 - np.dot(p1, p2) / (np.sqrt((p1 * p1).sum()) * np.sqrt((p2 * p2).sum()))

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)

x = np.random.rand(100, 8)
kmeans = KMeansClusterer(x, 10)
result, centers, distances, k_avg_dis = kmeans.cluster()
print(result)
print(centers)
print(distances)
print(k_avg_dis)
