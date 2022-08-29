#week3作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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

#待切分文本
sentence = "经常有意见分歧"
result = []
max_length = 0
for item in Dict:
    max_length = max(max_length, len(item))

sentence_length = len(sentence)
#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict, list):
    #TODO
    # print('list', len(list))
    new_str = ''.join(str(i) for i in list)
    # print(sentence, new_str)
    if len(sentence) == 0:
        # print('满足条件', list)
        result.append(list.copy())
        return

    if len(new_str) > sentence_length:
        return


    for i in range(max_length):
        if i > len(sentence) - 1:
            continue
        segment = sentence[: i + 1]
        if segment in Dict:
            list.append(segment)
            new_sentence = sentence[i + 1:]
            all_cut(new_sentence, Dict, list)
            list.pop()

all_cut(sentence, Dict, [])

for item in result:
    print(item)

#目标输出;顺序不重要
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