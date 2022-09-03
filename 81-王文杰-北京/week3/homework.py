#week3作业

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

maxKeyLen = 0
for (k,v) in Dict.items():
    if(len(k)>maxKeyLen):
        maxKeyLen = len(k)

#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    #TODO
    retList = list()
    isFind = False
    for j in range(0, maxKeyLen):
        if(j+1<=len(sentence)):
            preStr = sentence[0:j+1]
            if(preStr in Dict):
                isFind = True
                subList = list()
                subList.append(preStr)

                remainStr = sentence[j+1:]
                if(remainStr is not None and len(remainStr)>0):
                    remainList = list()
                    remainList = all_cut(remainStr, Dict)

                    for n in range(0, len(remainList)):
                        if(remainList[n] is not None):
                            retList.append(subList+remainList[n])
                else:
                    retList.append(subList)
    if(isFind == False):
        for m in range(0, len(retList)):
            retList[m].append(sentence[0:1])

    return retList

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

if __name__ == "__main__":
    result = all_cut(sentence, Dict)
    print(result)
