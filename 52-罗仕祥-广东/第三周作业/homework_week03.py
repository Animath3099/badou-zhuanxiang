#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2022/8/18 20:24
@Author  : luoshixiang
@Email   : just_rick@163.com
@File    : week03.py
@effect  : 实现基于词表的全切分
"""
import itertools


def get_dag(sentence, prefix_dict):
    DAG = {}
    N = len(sentence)
    for i in range(N):
        tmplist = []
        j = i
        frag = sentence[i]
        while j < N:
            if frag in prefix_dict:
                tmplist.append(j)
            j += 1
            frag = sentence[i:j + 1]
        if not tmplist:
            tmplist.append(i)
        DAG[i] = tmplist
    return DAG

def get_comb(DAG, sentence):

    # 每个列表抽取1个数据的所有组合
    tmplist = [v for k, v in DAG.items()]  # 取出字典中的值组成列表
    tmp_combination = []
    for comb in list(itertools.product(*tmplist)):
        tmp = []
        for j in range(len(comb)):
            if comb[j] not in tmp:
                if j == 0:
                    tmp.append(comb[j])
                elif comb[j] > comb[j - 1]:
                    tmp.append(comb[j])
                elif comb[j] <= comb[j - 1]:
                    del tmp[-1]
                    tmp.append(comb[j])
        if tmp not in tmp_combination:  # 对组合去重
            tmp_combination.append(tmp)

    comblist = []
    for i in range(len(tmp_combination)):
        g = tmp_combination[i]
        f = []
        for j in range(len(g)):
            if j == 0:
                f.append(sentence[:g[j] + 1])
            else:
                f.append(sentence[g[j - 1] + 1:g[j] + 1])
        comblist.append(f)

    word_list = [k for k,v in Dict.items()]
    # 剔除不存在的词组的组合
    combination = []
    for i in comblist:
        k = []
        for j in i:
            if j not in word_list:
                k = []
                break
            else:
                k.append(j)
        combination.append(k)
    target = [i for i in combination if i]     # 将空值剔除
    return target

def all_cut(sentence, Dict):

    # 1.DAG 获得有向无环图
    DAG = get_dag(sentence, Dict)
    # 2. 从DAG中输出所有可能的词汇
    target = get_comb(DAG, sentence)
    return target

if __name__ == '__main__':
    # 待切分文本
    sentence = "经常有意见分歧"
    Dict = {"经常": 0.1,
            "经": 0.05,
            "有": 0.1,
            "常": 0.001,
            "有意见": 0.1,
            "歧": 0.001,
            "意见": 0.2,
            "分歧": 0.2,
            "见": 0.05,
            "意": 0.05,
            "见分歧": 0.05,
            "分": 0.1}

    target = all_cut(sentence, Dict)
    for i in target:
        print(i)

