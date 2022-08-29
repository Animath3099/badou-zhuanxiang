
#sentence = input("input sentence you want to cut: ")
word_dic = {}
with open("./dict.txt", 'r', encoding='utf-8') as f:#取字典，只取名称和词性
    for lines in f:
        tmp_char = lines.strip().split(' ')
        word_dic[tmp_char[0]] = tmp_char[2]
#
def helper(start, s, ans):
    if start == len(sentence):
        ans.append(s[1:])
        return 
    tmp = s
    for i in range(max_window):
        if sentence[start:start+i+1] in word_dic:
            #print(tmp+","+sentence[start:start+i+1])
            helper(start+i+1, tmp+","+sentence[start:start+i+1], ans)
#找到字典中最大长度的词
def getMaxChar(word_dic):
    max_num = 0
    for i in word_dic:
        if len(i) >= max_num:
            max_num = len(i)
    return max_num
global max_window
max_window = getMaxChar(word_dic)
#主函数
def AllLengthCut(sentence, word_dic, ans):
    print(f"max_windows is {max_window}")
    s = ''
    helper(0,s,ans)
    #print(f"ans is {ans[1:]}")
    return ans
#AllLengthCut(sentence, word_dic, []) ##自输入使用
####test
sentence_list = ['我爱中华人民共和国','有兴趣的同学可以深入了解上面提到的应用领域','临时停电导致数万斤鱼因缺氧大量死亡', '汶川地震截肢女孩实现学医梦想'] 
for sentence in sentence_list:
    print("***************************************************************************************************************************************")
    res = AllLengthCut(sentence, word_dic, [])
    for i in list(set(res)):
        print(i)
    print(f"数量为 {len(res)}")
    print(f"数量为 {len(list(set(res)))}") ##