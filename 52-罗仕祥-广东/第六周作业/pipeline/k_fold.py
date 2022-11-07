#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2022/10/11 22:14
@Author  : luoshixiang
@Email   : just_rick@163.com
@File    : k_fold.py
@effect  : 
"""

from sklearn.model_selection import KFold
#
# four_label = {}
# kf = KFold(n_splits = 5, shuffle=True, random_state=0)
# for train_index, test_index in kf.split(data):
#     clt = model.fit(data[train_index], four_label[train_index])
#     curr_score = curr_score + clt.score(data[test_index], four_label[test_index])
#     print(clt.score(data[test_index], four_label[test_index]))
#
# avg_score = curr_score / 5
# print("平均准确率为：", avg_score)


#导入相关数据
data = ["a", "b", "c", "d", "e", "f"]
#设置分组这里选择分成3份。
kf = KFold(n_splits = 3, shuffle=True, random_state=0)
#查看分组结果
for train, test in kf.split(data):
    print("%s-%s" % (train, test))

