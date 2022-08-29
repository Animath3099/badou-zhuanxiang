#分词方法最大正向切分的第三种实现方式

import re
import time
import json

#加载词前后缀词典
#值为0代表是前缀
#值为1代表是一个词且这个词向后没有更长的词
#值为2代表是一个词，但是有比他更长的词
def load_prefix_word_dict(path):
    prefix_dict = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            word = line.split()[0]
            for i in range(1, len(word)):
                if word[:i] not in prefix_dict: #不能用前缀覆盖词
                    prefix_dict[word[:i]] = 0  #前缀
                if prefix_dict[word[:i]] == 1:
                    prefix_dict[word[:i]] = 2
            prefix_dict[word] = 1  #词
    return prefix_dict

#输入字符串和字典，返回词的列表
def cut_method3(string, prefix_dict):
    if string == "":
        return []
    words = []  # 准备用于放入切好的词
    start_index, end_index = 0, 1  #记录窗口的起始位置
    window = string[start_index:end_index] #从第一个字开始
    find_word = window  # 将第一个字先当做默认词
    while start_index < len(string):
        #窗口没有在词典里出现
        if window not in prefix_dict or end_index > len(string):
            words.append(find_word)  #记录找到的词
            start_index += len(find_word)  #更新起点的位置
            end_index = start_index + 1
            window = string[start_index:end_index]  #从新的位置开始一个字一个字向后找
            find_word = window
        #窗口是一个词,且不是任何词的前缀
        elif prefix_dict[window] == 1:
            words.append(window)  # 记录找到的词
            start_index += len(window)  # 更新起点的位置
            end_index = start_index + 1
            window = string[start_index:end_index]  # 从新的位置开始一个字一个字向后找
            find_word = window
        #窗口是一个词，但是有包含它的词，所以要再往后看
        elif prefix_dict[window] == 2:
            find_word = window  #查找到了一个词，还要在看有没有比他更长的词
            end_index += 1
            window = string[start_index:end_index]
        #窗口是一个前缀
        elif prefix_dict[window] == 0:
            end_index += 1
            window = string[start_index:end_index]
    #最后找到的window如果不在词典里，把单独的字加入切词结果
    if prefix_dict.get(window) != 1:
        words += list(window)
    else:
        words.append(window)
    return words

#cut_method是切割函数
#output_path是输出路径
def main(cut_method, input_path, output_path):
    word_dict = load_prefix_word_dict("dict.txt")
    writer = open(output_path, "w", encoding="utf8")
    start_time = time.time()
    with open(input_path, encoding="utf8") as f:
        for line in f:
            words = cut_method(line.strip(), word_dict)
            writer.write(" / ".join(words) + "\n")
    writer.close()
    print("耗时：", time.time() - start_time)
    return

string = "们审慎推测，由于人工智能联结主义基础下的神经网络模型有潜力契合自然语言的内在结构分解方式，"
prefix_dict = load_prefix_word_dict("dict.txt")
# print(json.dumps(prefix_dict, ensure_ascii=False, indent=2))
# words = cut_method3(string, prefix_dict)
# print(words)
main(cut_method3, "corpus.txt", "cut_method3_output.txt")
