# week3作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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

max_dic_len = max([len(dic) for dic in Dict])


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    seg_dict = dict()
    s_len = len(sentence)
    # 待切分句子遍历，找到所有可能切分位置
    for i in range(s_len + 1):
        tmp_list = list()
        for j in range(i + 1, min(i + max_dic_len + 1, s_len + 1)):
            word = sentence[i:j]
            if word in Dict:
                tmp_list.append(j)
        seg_dict[i] = tmp_list

    result_list = [[0, seg] for seg in seg_dict[0]]
    for i in range(1, s_len):
        seg_dict_iter(seg_dict, i, result_list)

    target = list()
    # 将切分路径转换为中文单词
    for result in result_list:
        result_target = [sentence[result[i]:result[i + 1]] for i in range(len(result) - 1)]
        target.append(result_target)

    return target


# 根据seg_dic, 得到全切分所有路径
def seg_dict_iter(seg_dict, start, result_list):
    curr_seg_list = seg_dict[start]
    result_len = len(result_list)
    # 遍历当前结果集数据
    for i in range(result_len):
        # 若结果序列的末尾与当前查看的切分位置香匹配，补全所有可能路径
        if result_list[i][-1] == start:
            for seg in curr_seg_list[1:]:
                result_list.append(result_list[i] + [seg])
            result_list[i] = result_list[i] + [curr_seg_list[0]]


# 目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]

if __name__ == "__main__":
    # 待切分文本
    sentence = "经常有意见分歧"
    target = all_cut(sentence, Dict)
    for tmp in target:
        print(tmp)
