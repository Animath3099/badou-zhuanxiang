#week4作业

'''

1.在kmeans聚类基础上，实现根据类内距离排序，输出结果

2.在不进行文本向量化的前提下对文本进行kmeans聚类(可采用jaccard距离)
'''

import jieba

#计算公共词
#前面1-的原因是使得文本相似是距离接近0，文本不同时文本接近1，更接近距离的概念
def jaccard_distance(list_of_words1, list_of_words2):
    return 1 - len(set(list_of_words1) & set(list_of_words2)) / len(set(list_of_words1) | set(list_of_words2))

a = jieba.lcut("今天真倒霉")
b = jieba.lcut("今天太走运了")
print(jaccard_distance(a, b))