# week3作业
import copy
from pprint import pprint
# 目标输出;顺序不重要
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


# 实现全切分
# 实现全切分函数，输出根据字典能够切分出的所有的切分方式

#寻找最长词
def maxLength(Dict):
    maxlength=0
    first={}
    key0=[]
    for key,value in Dict.items():
        long=len(key)
        if long>maxlength:
            maxlength=long
        key0.append(key[0])
    return maxlength,key0

#寻找切分中有多条路径的路径首词
def FirstWord(key):
    lenkey=len(key)
    first={}
    while lenkey>2:
        str=key[0]
        key=key[1:]
        count=1
        for pos in range(len(key)):
            if str==key[pos]:
                count+=1
                first[key[pos]]=count
        lenkey=len(key)
    return first

#最大切分
def all_Cut(sent,dic,first,maxlen):
    cut=[]
    j = 0
    for i in range(0, len(sent)):
        while j < maxlen:
            if i + j < len(sent):
                window = sent[i:i + j + 1]
            else:
                window = ''
            if window in dic.keys():
                #判断该字是否首字
                if window[0] in first.keys():
                    temp1=[]
                    temp3=[]
                    if len(cut) !=0:
                        for pos in range(len(cut)):
                            #判断window的首字是否在cut[pos]的最后一项中
                            if window[0] in cut[pos][-1]:
                                if len(cut[pos][-1]) != 1:
                                    pass
                                else:
                                    temp1 = copy.deepcopy(cut[pos])
                                    if len(temp1) !=1:
                                        del temp1[-1]
                                    else:
                                        temp1=[]
                                    temp1.append(window)
                                    # temp1[1]+=len(window)
                                    temp3.append(temp1)
                                    temp1=[]
                            else:
                                cut[pos].append(window)
                        if len(temp3)==0:
                            pass
                        else:
                            for poss in range(len(temp3)):
                                cut.append(temp3[poss])
                    else:#cut为空
                        cut.append([window])
                else:
                    for pos in range(len(cut)):
                        if window[0] not in ''.join(cut[pos]):
                            cut[pos]+=window
            j += 1
        j = 0
    # print(cut)
    return cut

if __name__ == "__main__":
    maxlenth,key=maxLength(Dict)
    first=FirstWord(key)
    result=all_Cut(sentence,Dict,first,maxlenth)
    pprint(result)





