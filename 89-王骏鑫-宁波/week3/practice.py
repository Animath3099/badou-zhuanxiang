# week3作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
# def all_cut(sentence, Dict):
#     #TODO
#     return target


# 正向最大匹配
def cut_method1(string, word_dict, max_len):

    words = []
    while string != "":
        lens = min(max_len, len(string))
        word = string[:lens]
        while word not in word_dict:
            if len(word) == 1:
                break
            word = word[:len(word) - 1]
        words.append(word)
        string = string[len(word):]
    words_count = 0
    for i in words:
        if len(i) != 1:
            words_count += 1
    return words, words_count, len(words)-words_count


# 反向最大匹配
def cut_method2(string, word_dict, max_len):
    words = []
    while string != "":
        lens = min(max_len, len(string))
        word = string[-lens:]
        while word not in word_dict:
            if len(word) == 1:
                break
            word = (word[-len(word) + 1:])
        words.append(word)
        string = (string[-(len(string)):-len(word)])
    words.reverse()
    words_count = 0
    for i in words:
        if len(i) != 1:
            words_count += 1
    return words, words_count, len(words)-words_count


# 双向最大匹配
def cut_method3(string, words_dict, max_len):
    words1, words_num1, single1 = cut_method1(string, words_dict, max_len)
    words2, words_num2, single2 = cut_method2(string, words_dict, max_len)
    if words_num1 == words_num2:
        if single1 > single2:
            return words2
        elif single1 < single2:
            return words1
        elif single1 == single2:
            return


def all_cut(sentence, Dict):
    if sentence == "":
        return []
    target = []
    words_dict = {word: len(word) for word in Dict.keys()}
    print(sentence)
    print(words_dict)
    print("-" * 30)
    # 含单字词分割
    target.append(cut_method1(sentence, words_dict.keys(), max(words_dict.values()))[0])
    target.append(cut_method2(sentence, words_dict.keys(), max(words_dict.values()))[0])
    # 不含单字词分割
    list1 = []
    for key, val in words_dict.items():
        if val == 1:
            list1.append(key)
    for i in list1:
        del words_dict[i]
    print(words_dict)
    target.append(cut_method1(sentence, words_dict.keys(), max(words_dict.values()))[0])
    target.append(cut_method2(sentence, words_dict.keys(), max(words_dict.values()))[0])
    cut_method3(sentence, words_dict.keys(), max(words_dict.values()))
    return target


# print(all_cut(sentence, Dict))


def all_cuts(sentence, Dict):
    # 第一步：全可能切分
    import random
    max_len = len(max(list(Dict.keys())))
    words_list = []
    for i in range(2000):
        sen = sentence
        words = []
        while sen != '':
            length = random.randint(1, max_len)
            try:
                word = sen[:length]
                words.append(word)
                sen = sen[len(word):]
            except:
                if length > len(sen):
                    length = random.randint(1, len(sen))
                    word = sen[:length]
                    words.append(word)
                    sen = sen[len(word):]
                else:
                    print("None")
        words_list.append(words)
    word_list = []
    for j in words_list:
        if j not in word_list:
            word_list.append(j)

    # 第二步：与词表进行匹配可行性
    target = []
    for List in word_list:
        flag = True
        for i in List:
            if i not in Dict.keys():
                flag = False
                break
        if flag != False:
            target.append(List)
    return target


possible = all_cuts(sentence, Dict)
print(len(possible), possible)


"""
统计词数并对比，若相同则对比单字词，再相同则对比非字典词
"""

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
