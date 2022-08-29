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


# 根据最大词长度窗口计算正向最大匹配
def forward_max_cut(sentence, Dict, max_length_word):
    words = []
    while sentence != '':
        stride = min(len(sentence), max_length_word)
        word = sentence[:stride]
        while word not in Dict:
            if len(word) == 1:
                break
            word = word[:len(word)-1]
        words.append(word)
        sentence = sentence[len(word):]

    return words


# 根据最大词长度窗口计算反向最大匹配
def backward_max_cut(sentence, Dict, max_length_word):
    words = []
    while sentence != '':
        stride = min(len(sentence),  max_length_word)
        word = sentence[len(sentence) - stride:]
        while word not in Dict:
            if word == 1:
                break
            word = word[1-len(word):]
        words.insert(0, word)
        sentence = sentence[:len(sentence) - len(word)]
    return words

# 分别进行正反向最大匹配,然后再各自进行全切分


def all_cut(sentence, Dict):
    target = []
    words = []
    max_length_word = 0
    for word in Dict:
        max_length_word = max(len(word), max_length_word)  # 计算单词的最大长度

    part_cut = []
    part_cut.append(forward_max_cut(sentence, Dict, max_length_word))
    part_cut.append(backward_max_cut(sentence, Dict, max_length_word))

    # 正反方向最大匹配结果是否相等,相等取之一
    if part_cut[0] == part_cut[1]:
        part_cut.pop()

    for k in range(len(part_cut)):
        target.append([part_cut[k]])
        for id, word in enumerate(target[k][0]):
            len_target = len(target[k])
            if len(word) > 1:
                word_part1 = word[0]
                start_index_wp1 = 0
                end_index_wp1 = start_index_wp1 + 1
                start_index_wp2 = 1
                end_index_wp2 = start_index_wp2 + 1
                while word_part1 != '' and len(word_part1) < len(word):
                    if word_part1 in Dict:
                        for step in range(len(word)-len(word_part1)):
                            words = []
                            words.append(word_part1)
                            start_index_wp2 = len(word_part1)
                            end_index_wp2 = start_index_wp2 + step + 1

                            while start_index_wp2 < len(word) and end_index_wp2 <= len(word):
                                word_part2 = word[start_index_wp2:end_index_wp2]
                                if word_part2 in Dict:
                                    words.append(word_part2)
                                    start_index_wp2 += len(word_part2)
                                else:
                                    end_index_wp2 += 1

                            if len(words) >= 2:
                                for r in range(len_target):
                                    words_cat = []
                                    j = target[k][r].index(word)
                                    words_cat = target[k][r][:j] + \
                                        words + target[k][r][j+1:]
                                    if k == 0:
                                        target[k].append(words_cat)
                                    # 判断对反向最大匹配的结果进行切分得到的结果是否存在正向最大匹配结果中
                                    elif words_cat not in target[0]:
                                        target[k].append(words_cat)

                        end_index_wp1 += 1
                        start_index_wp2 = end_index_wp1
                        end_index_wp2 = start_index_wp2 + 1
                        word_part1 = word[start_index_wp1:end_index_wp2]
                    else:
                        end_index_wp2 += 1
                        if end_index_wp2 < len(word):
                            word_part1 = word[start_index_wp1:end_index_wp2]
                            start_index_wp2 = len(word_part1)
                            end_index_wp2 = start_index_wp2 + 1
                        else:
                            break

    target = target[0] + target[1]
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
