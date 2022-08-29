"""输出一句话的所有可能的分词方式
使用递归，一个类存储一种词典
"""


class AllPossibleSeparation:

    def __init__(self, path):
        self.word_dict, self.max_word_length = self.load_word_dict(path)
        self.word_dict = set(self.word_dict.keys())
        self.res = []

    # 加载词典
    @staticmethod
    def load_word_dict(path):
        max_word_length = 0
        word_dict = {}  # 用set也是可以的。用list会很慢
        with open(path, encoding="utf8") as f:
            for line in f:
                word = line.split()[0]
                word_dict[word] = 0
                max_word_length = max(max_word_length, len(word))
        return word_dict, max_word_length

    def separate_one_sentence(self, sentence):
        if not sentence:
            return []

        self.res = []
        self.recursion([], sentence)
        return self.res

    def recursion(self, pre_list, rest_sentence):
        if not rest_sentence:
            self.res.append(pre_list)
            return

        for i in range(min(len(rest_sentence), self.max_word_length), 1, -1):
            if rest_sentence[:i] in self.word_dict:
                self.recursion(pre_list + [rest_sentence[: i]], rest_sentence[i:])

        self.recursion(pre_list + [rest_sentence[0]], rest_sentence[1:])


separation_1 = AllPossibleSeparation("dict.txt")

string_1 = "测试字符串"

print(separation_1.separate_one_sentence(string_1))