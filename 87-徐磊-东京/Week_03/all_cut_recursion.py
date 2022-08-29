# week3作业
'''
实现全切分函数，输出根据字典能够切分出的所有的切分方式
'''

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


# 从单个字符开始切分
def all_cut(sentence, Dict):
    target = []

    for id in range(1, len(sentence) + 1):
        word = sentence[0:id]
        if word in Dict:
            # 单词和句子的长度相同时,表明切分已到句子的最后一个字符,返回切分结果
            if len(word) == len(sentence):
                target.append([word])
                return target
            else:
                sub_segment = all_cut(sentence[id:], Dict)
                # 将word所有可能的组合切分分别加入到word所在的列表中
                for seg in sub_segment:
                    seg = [word] + seg
                    target.append(seg)
    return target


# 输出切分结果
target = all_cut(sentence, Dict)
print()
for i in range(len(target)):
    print(target[i])

target_answer = [
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


# 判断切分结果是否与参考答案一致
answer_check = [item in target_answer for item in target]
if answer_check.count(True) == len(target_answer):
    print('\n全切分结果与参考答案一样\n')
else:
    print('\n全切分结果与参考答案不一样\n')
