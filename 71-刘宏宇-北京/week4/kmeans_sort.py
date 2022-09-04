import jieba
import numpy as np
from math import sqrt
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

'''
实现kmeans的类内距离计算
'''

def load_sentence(path):
    '''
    加载所有标题
    '''
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip().replace(' ', '')
            sentences.add(" ".join(jieba.lcut(sentence)))
        f.close()

    print("获取句子数量：", len(sentences))
    return sentences

def sentences_to_vectors(sentences, model):
    '''
    标题向量化：词向量取平均
    '''
    vectors = []
    for sentence in sentences:
        # 标题向量, 类型 <class 'numpy.ndarray'>
        vector = np.zeros(model.vector_size)

        words = sentence.split()
        for word in words:
            try:
                # 向量加法
                vector += model.wv[word] 
            except KeyError:
                vector += np.zeros(model.vector_size)
        # 向量除法
        vector = vector / len(words)

        vectors.append(vector)  
    
    # <class 'list'> --> <class 'numpy.ndarray'>
    return np.array(vectors)

def test_KMeans_1():
    '''
    测试KMeans聚类，标题向量聚类
    '''
    model = Word2Vec.load("../model.w2v") # 加载词向量模型-Word2Vec<vocab=19322, vector_size=100, alpha=0.025>
    sentences = load_sentence("../titles.txt")
    vectors = sentences_to_vectors(sentences, model)
    
    n_clusters = int(sqrt(len(vectors))) # 指定聚类数量, 为样本数算术平方根
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    cluster_centers = kmeans.cluster_centers_ # 获取聚类中心，(42, 100)

    # defaultdict 是 dict 的一个子类，可以为不存在的键返回一个默认值，而不会引发 keyerror 异常
    # defaultdict 在初始化时可以提供一个 default_factory 的参数，default_factory 接收一个工厂函数作为参数， 可以是 int、str、list 等内置函数，默认都为空值
    
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        #同标签的放到一起
        sentence_label_dict[label].append(sentence)

    vector_label_dict = defaultdict(list)
    for vector, label in zip(vectors, kmeans.labels_):
        #同标签的放到一起
        vector_label_dict[label].append(vector)

    for i, center in enumerate(cluster_centers):
        print(f"cluster {i} 类内距离排序：")

        vertor_dists = {}
        for j, vertor in enumerate(vector_label_dict[i]):
            dist = com_distances(vertor, center, 'cosine')
            vertor_dists[j] = dist
        vertor_dists = dict(sorted(vertor_dists.items(), key=lambda x:x[1], reverse=True))

        for j, key in enumerate(vertor_dists.keys()):
            if j < 10:
                print(sentence_label_dict[i][key].replace(" ", ""))

        # print(f"cluster {i} 类内距离不排序：")

        # sentences = sentence_label_dict[i]
        # for j in range(min(10, len(sentences))):
        #     print(sentences[j].replace(" ", ""))

        print("---------")

def test_KMeans_2():
    '''
    测试KMeans聚类
    '''
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 0], [4, 4],
                  [4, 5], [0, 1], [2, 2],
                  [3, 2], [5, 5], [1, -1]])
    
    #定义一个kmeans计算类
    kmeans = KMeans(2)
    #进行聚类计算
    kmeans.fit(X)

    # print(kmeans.cluster_centers_)
    # print(kmeans.labels_)

    vector_label_dict = defaultdict(list)
    for vector, label in zip(X, kmeans.labels_):
        #同标签的放到一起
        vector_label_dict[label].append(vector)

    for i, center in enumerate(kmeans.cluster_centers_):
        print(f"cluster{i}：类中心：{center}，类内距离排序")
        
        vertor_dists = {}
        for j, vertor in enumerate(vector_label_dict[i]):
            dist = com_distances(vertor, center, 'cosine')
            vertor_dists[j] = dist

        vertor_dists = dict(sorted(vertor_dists.items(), key=lambda x:x[1], reverse=True))

        for key in vertor_dists.keys():
            print(vector_label_dict[i][key])


def com_distances(x, y, way):
    '''
    输入数据为一维向量, 计算向量余弦距离和欧氏距离
    np.multiply() 元素乘法，对应位置元素相乘；np.dot() 矩阵乘法，点积
    '''
    # x = np.array([3, 2])
    # y = np.array([5, 5])

    if way == 'euclidean':
        # 欧氏距离
        z = x - y
        dist = sqrt(np.dot(z, z)) # dist = sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))
        return dist
    if way == 'cosine':
        # 余弦距离
        similiarity = np.dot(x, y) / sqrt(np.dot(x, x) * np.dot(y, y))
        dist = 1 - similiarity
        return dist

def norm(x):
    '''
    标准化
    '''
    if x.ndim == 1:
        x_norm = np.multiply(x, x).sum(axis=0)
    
    if x.ndim == 2:
        x_norm = np.multiply(x, x).sum(axis=1)

    return x_norm

def cosine_distance(a, b):
    '''
    输入数据为二维向量
    '''
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    
    similiarity = np.dot(a, b.T)/(a_norm * b_norm) 
    dist = 1. - similiarity
    return dist


if __name__ == '__main__':

    test_KMeans_1()
