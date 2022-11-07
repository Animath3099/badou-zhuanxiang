import json
import copy
from ngram_language_model import NgramLanguageModel
"""
文本纠错demo
加载同音字字典
加载语言模型
基本原理：
对于文本中每一个字，判断在其同音字中是否有其他字，在替换掉该字时，能使得语言模型计算的成句概率提高
"""


class Corrector:
    def __init__(self, language_model):
        #语言模型
        self.language_model = language_model
        #候选字字典
        self.sub_dict = self.load_tongyinzi("tongyin.txt")
        #成句概率的提升超过阈值则保留修改
        self.threshold = 7

    #实际上不光是同音字，同形字等也可以加入，本质上是常用的错字
    def load_tongyinzi(self, path):
        tongyinzi_dict = {}
        with open(path, encoding="utf8") as f:
            for line in f:
                char, tongyin_chars = line.split()
                tongyinzi_dict[char] = list(tongyin_chars)
        return tongyinzi_dict

    #根据替换字逐句计算成句概率的提升值
    def get_candidate_sentence_prob(self, candidates, char_list, index):
        if candidates == []:
            return [-1]
        result = []
        for char in candidates:
            char_list[index] = char
            sentence = "".join(char_list)
            sentence_prob = self.language_model.predict(sentence)
            #减去基线值，得到提升了多少
            sentence_prob -= self.sentence_prob_baseline
            result.append(sentence_prob)
        return result

    #纠错逻辑
    def correction(self, string):
        char_list = list(string)
        fix = {}
        #计算一个原句的成句概率
        self.sentence_prob_baseline = self.language_model.predict(string)
        for index, char in enumerate(char_list):
            candidates = self.sub_dict.get(char, [])
            #注意使用char_list的拷贝，以免直接修改了原始内容
            candidate_probs = self.get_candidate_sentence_prob(candidates, copy.deepcopy(char_list), index)
            #如果成句概率的提升大于一定阈值，则记录替换结果
            if max(candidate_probs) > self.threshold:
                #找到最大成句概率对应的替换字
                sub_char = candidates[candidate_probs.index(max(candidate_probs))]
                print("第%d个字建议修改：%s -> %s, 概率提升： %f" %(index, char, sub_char, max(candidate_probs)))
                fix[index] = sub_char
        #替换后字符串
        char_list = [fix[i] if i in fix else char for i, char in enumerate(char_list)]
        return "".join(char_list)


corpus = open("财经.txt", encoding="utf8").readlines()
lm = NgramLanguageModel(corpus, 3)

cr = Corrector(lm)
string = "每国货币政册空间不大"
fix_string = cr.correction(string)
print("修改前：", string)
print("修改后：", fix_string)