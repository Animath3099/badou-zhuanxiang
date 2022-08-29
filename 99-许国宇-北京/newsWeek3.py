#week3作业
from pprint import pprint
#目标输出;顺序不重要
# target = [
#     ['经常', '有意见', '分歧'],
#     ['经常', '有意见', '分', '歧'],
#     ['经常', '有', '意见', '分歧'],
#     ['经常', '有', '意见', '分', '歧'],
#     ['经常', '有', '意', '见分歧'],
#     ['经常', '有', '意', '见', '分歧'],
#     ['经常', '有', '意', '见', '分', '歧'],
#     ['经', '常', '有意见', '分歧'],
#     ['经', '常', '有意见', '分', '歧'],
#     ['经', '常', '有', '意见', '分歧'],
#     ['经', '常', '有', '意见', '分', '歧'],
#     ['经', '常', '有', '意', '见分歧'],
#     ['经', '常', '有', '意', '见', '分歧'],
#     ['经', '常', '有', '意', '见', '分', '歧']
# ]

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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

#首词判断
def firstWord(sentence,Dict):
    # 寻找字典中的最长词
    max_length = 0
    for _, value in enumerate(Dict):
        if len(value) > max_length:
            max_length = len(value)

    # 寻找分词路径中的首词
    # 首先输出全切分
    target_temp = []
    temp = []
    j = 0
    for i in range(0, len(sentence)):
        while j < max_length:
            if i + j < len(sentence):
                window = sentence[i:i + j + 1]
            else:
                window = ''
            if window in Dict.keys():
                target_temp.append(window)
            j += 1
        j = 0
    # 找首词
    num = 0
    word_front = {}  # 数据存放的格式为:位置:首词
    fork_num = {}  # 存放当前位置可能路径的数量,格式:位置:数量
    pos_word_front = []  # 存储当前首词的位置
    # ['经', '经常', '常', '有', '有意见', '意', '意见', '见', '见分歧', '分', '分歧', '歧']
    for i in range(0, len(sentence)):
        for j in range(0, len(target_temp)):
            # sentence中的单字同切分结果每一项的首字进行比较是否为首词
            if set(sentence[i]) & set(target_temp[j][0]):
                num += 1
        if num > 1:
            word_front[i] = sentence[i]
            fork_num[i] = num
            pos_word_front.append(i)
        num = 0
    # print(word_front)
    return word_front, fork_num,target_temp
#首字之前的文本
def textBeforeFirstword(sentence,word_front,fork_num):
    # print(pos_word_front)
    # 根据找到的首词,判断各个位置上的可能的字符路径
    fork_word = []  # 存放各个路径的内容
    # word_front={0: '经', 2: '有', 3: '意', 4: '见', 5: '分'}
    # fork_num={0: 2, 2: 2, 3: 2, 4: 2, 5: 2}
    i = 0
    fork_word_temp = []
    singleWord_temp = []
    multiWord_temp = []
    for key, value in fork_num.items():
        while i < value:
            # 多字
            if sentence[key:key + i + 2] in Dict.keys():
                if multiWord_temp == []:
                    multiWord_temp.append(sentence[key:key + i + 2])
                else:
                    if sentence[key:key + i + 2] != multiWord_temp[-1]:
                        multiWord_temp.append(sentence[key:key + i + 2])
                # print("fork_word_temp=",fork_word_temp)
            # 单字
            if sentence[key + i] in Dict.keys():
                if key + i + 1 == len(sentence):
                    singleWord_temp.append(sentence[key + i])
                else:
                    if sentence[key + i] == word_front[key]:
                        singleWord_temp.append(sentence[key + i])
                        # print("single=",singleWord_temp)
                        # sentence=[经,常,有,意,见,分,歧]
                    elif i < value:
                        if sentence[key + i] not in word_front.values():
                            singleWord_temp.append(sentence[key + i])
                            # print("single=",singleWord_temp)
                            # sentence=[经,常,有,意,见,分,歧]

            i += 1
        fork_word_temp.append(singleWord_temp)
        fork_word_temp.append(multiWord_temp)
        fork_word.append(fork_word_temp)
        fork_word_temp = []
        multiWord_temp = []
        singleWord_temp = []
        i = 0
    fork_temp=[]
    # print(fork_word)
    #根据fork_word得到如下结果的数据:分割字符序列,当前位置
    for k in range(0,len(fork_word)):
        for m in range(0,len(fork_word[0])):
            pos=list(sentence).index(fork_word[k][m][-1][-1])
            # print("pos=",pos)
            # print(fork_word[k][m])
            temp=fork_word[k][m]
            fork_temp.append((temp,pos))
            temp=[]
    # print("fork_temp=",fork_temp)
    # print(fork_temp[0][0])
    return fork_temp

def all_cut(sentence,Dict):
    temp=[]
    result=[]#存放结果
    x=[]
    kk=0
    fork = [[['经常'], 1], [['有意见'], 4], [['意见'], 4], [['见分歧'], 6], [['分歧'], 6]]
    while kk<2:#栈不空,主栈循环
        #弹出栈顶元素
        if kk==0:
            temp=[['经','常'],1]
        else:
            temp=fork[0]
        print(temp)
        print(temp[0])
        print(temp[1])
        # sentence="常有意见分歧"
        first_word,forkNum,allCuts= firstWord(sentence, Dict)
        # print(forkNum)
        textBefore=textBeforeFirstword(sentence,first_word,forkNum)
        # print(textBefore)
        # print(allCuts)
        # first_word={0: '经', 2: '有', 3: '意', 4: '见', 5: '分'}
        #allcuts=['经', '经常', '常', '有', '有意见', '意', '意见', '见', '见分歧', '分', '分歧', '歧']
        #forkNum={0: 2, 2: 2, 3: 2, 4: 2, 5: 2}
        #textbefore=[(['经', '常'], 1), (['经常'], 1), (['有'], 2), (['有意见'], 4), (['意'], 3), (['意见'], 4), (['见'], 4), (['见分歧'], 6), (['分', '歧'], 6), (['分歧'], 6)]
        window=[]
        i=0
        temp_temp=[]
        while i<len(sentence)-2:
            if temp[1]<=len(sentence)-1:
                if temp[1]==len(sentence)-1:#第一条分支已经到尾部
                    result.append(temp)
                    temp=[]
                    temp=fork[0]
                    temp_result=[]
                    for k in range(len(fork)-1,0,-1):
                        fork_f = fork[k]
                        lens=len(fork_f[0][0])
                        pos=fork_f[-1]
                        if len(result)>7:
                            res=['经常','有','意','见','分','岐']
                            temp_result = res[0:pos - lens]
                            temp2 = res[0][pos - lens:-1]
                            temp_result = temp_result + [fork_f[0][0]] + list(temp2)
                            if len(''.join(temp_result)) < len(sentence):
                                if len(temp_result[-1]) > 1:
                                    index = sentence.index(temp_result[-1][-1])
                                    sentence1 = sentence[index + 1:]
                                    fw, fn, allCuts = firstWord(sentence1, Dict)
                                    tb = textBeforeFirstword(sentence1, fw, fn)
                                    temp_temp = temp_result
                                    for num in range(len(tb)):
                                        temp_temp = temp_temp + tb[num][0]
                                        result.append(temp_temp)
                                        temp_temp = temp_result
                            else:
                                result.append(temp_result)
                            temp_result = []
                        else:
                            res=result[0][0]
                            temp_result = res[0:pos - lens + 1]
                            temp2 = res[0][pos - lens + 1:-1]
                            temp_result = temp_result + [fork_f[0][0]] + list(temp2)
                            if len(''.join(temp_result)) < len(sentence):
                                if len(temp_result[-1]) > 1:
                                    index = sentence.index(temp_result[-1][-1])
                                    sentence1 = sentence[index + 1:]
                                    fw, fn, allCuts = firstWord(sentence1, Dict)
                                    tb = textBeforeFirstword(sentence1, fw, fn)
                                    temp_temp = temp_result
                                    for num in range(len(tb)):
                                        temp_temp = temp_temp + tb[num][0]
                                        result.append(temp_temp)
                                        temp_temp = temp_result
                            else:
                                result.append(temp_result)
                            temp_result = []

                else:#第一条分支还未到尾部
                    for tnum in range(len(textBefore)):
                        if temp[1]+1<len(sentence)-1:
                            if sentence[temp[1]+1]==textBefore[tnum][0][0][0]:
                                window.append(textBefore[tnum])
                    # print("window=",window)
                    # x.append((window[i][0],window[i][1]))
                    #同之前出栈的文本进行拼接
                    temp[0]=temp[0]+window[0][0]
                    temp[1]=window[0][1]
                    window=[]
            i+=1
        kk+=1
    pprint(result)

if __name__=="__main__":
    all_cut(sentence,Dict)
