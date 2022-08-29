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

# 待切分文本
sentence = "经常有意见分歧"

# 实现全切分函数，输出根据字典能够切分出的所有的切分方式


def all_cut(sentence, Dict):
    for i in range(len(sentence)):
        DAG = []
        n = min(4+i, len(sentence)+1)
        for j in range(i, n):
            if sentence[i:j] in Dict:
                DAG.append(sentence[i:j])
        DAG_list.append(DAG)

    target = []
    for i in range(len(DAG_list[0])):  # 第一个字 组成词列表中
        len0 = len(DAG_list[0][i])
        new = []
        if len0 >= len(DAG_list):
            new.append([DAG_list[0][i]])
        else:
            for j in range(len(DAG_list[len0])):  # 词组长度对应的 下个字 组成词列表中
                print('i:', i)
                print('j:', i)
                print('DAG_list[0][i]:', DAG_list[0][i])
                print('DAG_list[len0][j]:', DAG_list[len0][j])
                new.append([DAG_list[0][i]]+[DAG_list[len0][j]])
        target.extend(new)

    for i in range(len(DAG_list)):
        target = creat_last(target)  # 调用了ioi函数
    return target


def creat_last(target):  # 调用DAG，以上一个list为基础生成下个list
    new_list = []
    for i in range(len(target)):
        s = ''
        for j in range(len(target[i])):
            s += target[i][j]
        len2 = len(s)
        new = []
        if len2 >= len(DAG_list):
            new.append(target[i])
        else:
            for k in range(len(DAG_list[len2])):  # 词组长度对应的 下个字 组成词列表中
                new.append(target[i]+[DAG_list[len2][k]])
        new_list.extend(new)
    return new_list


DAG_list = []
print(all_cut(sentence, Dict))


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
