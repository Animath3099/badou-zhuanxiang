from pyhanlp import *


def load_dictionary():
    IOUtil = JClass('com.hankcs.hanlp.corpus.io.IOUtil')
    path = HanLP.Config.CoreDictionaryPath.replace('.txt', '.mini.txt')
    dic = IOUtil.loadDictionary([path])
    return set(dic.keySet())


def fully_segment(text, dic):
    list = []
    for i in range(len(text)):
        for j in range(i + 1, len(text) + 1):
            temp = text[i:j]
            if temp in dic:
                list.append(temp)
    return list


if __name__ == "__main__":
    dic = load_dictionary()
    print(fully_segment("经常有意见分歧", dic))
