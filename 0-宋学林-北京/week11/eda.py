import jieba
import random
from collections import defaultdict


random.seed(2022)

# 加载停用词列表
stopwords = set()
with open('stopwords.txt', encoding='utf8') as f:
    for stop_word in f.readlines():
        stopwords.add(stop_word.strip())         # 一维列表，元素为停用词

# 加载同义词表
synonyms = defaultdict(list)
with open("synonym.txt", encoding="utf8") as f:
    for line in f:
        words = line.strip().split()
        for word in words:
            synonyms[word] += [w for w in words if w != word]



########################################################################
# 同义词替换，替换一个语句中的n个单词为其同义词
# 先过滤停用词，在剩余词中采样出n个，逐个替换
########################################################################
def synonym_replacement(words, n):
    #过滤停用词
    candidates = []
    for i, word in enumerate(words):
        if word not in stopwords and word in synonyms:
            candidates.append([i, word])
    #选取n个词做替换，n不能超过总数
    to_subs = random.sample(candidates, min(n, len(candidates)))
    #逐个替换
    for i, word in to_subs:
        sub_word = random.choice(synonyms[word])
        words[i] = sub_word
    return words


########################################################################
# 随机插入，随机在语句中插入n个词
# 与替换思路相近，最后一步改为随机插入，而不是替换
########################################################################
def random_insertion(words, n):
    # 过滤停用词
    candidates = []
    for i, word in enumerate(words):
        if word not in stopwords and word in synonyms:
            candidates.append([i, word])
    # 选取n个词做插入，n不能超过总数
    to_ins = random.sample(candidates, min(n, len(candidates)))
    #逐个替换
    for i, word in to_ins:
        new_word = random.choice(synonyms[word])
        position = random.choice(range(len(words))) #随机一个插入位置
        words = words[:position] + [new_word] + words[position:]
    return words


########################################################################
# 随机交换。随机交换句子中的两个单词n次
########################################################################

def random_swap(words, n):
    for i in range(n):
        a, b = random.sample(range(len(words)), 2)
        words[a], words[b] = words[b], words[a]
    return words

########################################################################
# 随机删除，以概率p删除语句中的词
# 先替换为none，在整体去除，因为在循环中修改列表长度是个不好的操作
########################################################################
def random_deletion(words, p):
    for i, word in enumerate(words):
        if random.random() < p:
            words[i] = None
    words = [w for w in words if w is not None]
    return words


########################################################################
# EDA函数
# alpha 和 文本长度 共同决定增强操作次数
# num_sample 加强得到几个样本
##########
def eda(sentence, alpha=0.15, num_sample=5):
    words = jieba.lcut(sentence)
    n = int(alpha * len(words))
    samples = []
    for i in range(num_sample):
        aug_method = random.choice([random_swap, random_deletion, random_insertion, synonym_replacement])
        aug_words = aug_method(words[:], n)  #注意使用原句的复制，不要使用原句，否则造成的修改会带到下一轮
        samples.append(["".join(aug_words), aug_method.__name__])
    return samples


if __name__ == '__main__':
    # 测试用例
    aug_sentences = eda(sentence="我们就像蒲公英，我也祈祷着能和你飞去同一片土地")
    print(aug_sentences)