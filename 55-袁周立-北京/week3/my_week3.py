import numpy as np


def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf-8") as f:
        for line in f:
            vocab[line.split(" ")[0]] = line.split(" ")[1]
    return vocab


def all_segmentation_method(vocab, sentence):
    word_matrix = np.zeros((len(sentence), len(sentence)))
    for i in range(len(sentence)):
        for j in range(i, len(sentence)):
            if sentence[i: j+1] in vocab.keys():
                word_matrix[i][j] = vocab[sentence[i: j+1]]
    result = matrix_segmentation(0, sentence, word_matrix)
    print(result)

    str = ""
    segmentation_index, sentenct2, score = result
    for i in range(len(sentenct2)):
        str += sentenct2[i]
        if i in segmentation_index:
            str += " / "
    print(str)

    return result


def matrix_segmentation(begin_index, sentence, word_matrix):
    if word_matrix.shape != (len(sentence), len(sentence)):
        raise Exception("矩阵维度错误")
    if len(sentence) == 0:
        return [], "", 0
    if len(sentence) == 1:
        return [], sentence, word_matrix[0][0]
    result = []
    for i in range(len(sentence)):
        sub_sentenct1, score1 = sentence[:i+1], word_matrix[i][i]
        segmentation_index, sub_sentenct2, score2 = matrix_segmentation(begin_index + i + 1, sentence[i+1:], word_matrix[i+1:, i+1:])
        result.append([[begin_index + i] + segmentation_index, sub_sentenct1 + sub_sentenct2, score1 + score2])
    result.sort(key=lambda x: x[2], reverse=True)
    return result[0]


def main(vocab_path, sentence):
    vocab = build_vocab(vocab_path)
    all_segmentation_method(vocab, sentence)


if __name__ == "__main__":
    vocab_path = "./week3 中文分词和tfidf特征应用/week3 中文分词和tfidf特征应用/上午-中文分词/dict.txt"
    sentence = "东京商品交易所橡胶期货也强势上扬"
    main(vocab_path, sentence)
