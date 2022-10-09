import math
import torch

from lm_train import LanguageModel
from loader import build_vocab

"""
文本纠错demo
加载同音字字典
加载语言模型
基本原理：
对于文本中每一个字，判断在其同音字中是否有其他字，在替换掉该字时，能使得语言模型计算的成句概率提高
"""

def load_trained_language_model(model_path):
    '''
    加载语言模型
    '''
    char_dim = 128        #每个字的维度,与训练时保持一致
    window_size = 6       #样本文本长度,与训练时保持一致

    # 建立字表
    vocab = build_vocab("vocab.txt")
    # 建立模型
    model = LanguageModel(vocab, char_dim)
    # 加载训练的模型权重
    model.load_state_dict(torch.load(model_path))
    # 添加所需参数
    model.window_size = window_size
    model.vocab = vocab

    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    return model

def load_tongyin(data_path):
    '''
    加载同音字
    '''
    tongyin_dict = {}
    with open(data_path, encoding="utf8") as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            key = line[0]
            if key not in tongyin_dict:
                tongyin_dict[key] = list(line[1])
        f.close()

    return tongyin_dict

def cal_perplexity(sentence, model):
    '''
    基于语言模型计算ppl, 模型给出预测的概率分布，获取相应字的概率作为当前成句概率
    '''
    prob = 0
    with torch.no_grad():
        # 末尾字符索引，最小为1，最大为句子长度减1
        for end in range(1, len(sentence)):
            start = max(0, end - model.window_size)

            window = sentence[start:end]
            x = [model.vocab.get(char, model.vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])          # 二维数据 torch.size(1, 128)        
            
            target = sentence[end]
            target_index = model.vocab.get(target, model.vocab["<UNK>"])

            if torch.cuda.is_available():
                x = x.cuda()

            pred_prob_distribute = model(x)[0] # 二维数据 torch.size(1, 4063), [0]表示取出当前的概率分布
            target_prob = pred_prob_distribute[target_index]

            # pred_index = torch.argmax(pred_prob_distribute) # 获取预测字符
            # print(sentence, window, target, model.index2char[int(pred_index)])

            prob += math.log(target_prob, 10)  # 计算对数概率

        return prob

def t_correct_based_lm(corrector, sentence):
    '''
    基于语言模型的文本纠错
    '''
    sentence_prob_baseline = cal_perplexity(sentence, corrector)
    print(f"原句子概率：{sentence_prob_baseline}")

    threshold = 0

    # 建立同音字
    vocab = load_tongyin("tongyin.txt")

    fix = {}

    # # 遇到同音字就替换，并用语言模型判断成句概率
    for index, char in enumerate(sentence):
        if char in vocab:
            char_list = list(sentence)

            candidate_probs = []
            for word in vocab[char]:
                char_list[index] = word
                tmp = ''.join(char_list)
                sentence_prob = cal_perplexity(tmp, corrector)
                #减去基线值，得到提升了多少
                sentence_prob -= sentence_prob_baseline
                candidate_probs.append(sentence_prob)
            
            if max(candidate_probs) > threshold:
                #找到最大成句概率对应的替换字
                sub_char = vocab[char][candidate_probs.index(max(candidate_probs))]
                print("第%d个字建议修改：%s -> %s, 概率提升： %f" %(index, char, sub_char, max(candidate_probs)))
                fix[index] = sub_char

        else:
            continue

    char_list = [fix[i] if i in fix else char for i, char in enumerate(list(sentence))]
    return "".join(char_list)

if __name__ == "__main__":
    string = "每国货币政册空间不大"

    corrector = load_trained_language_model("data/财经.pth")
    fix_string = t_correct_based_lm(corrector, string)

    print("修改前：", string)
    print("修改后：", fix_string) #美国货币政策空间不大
