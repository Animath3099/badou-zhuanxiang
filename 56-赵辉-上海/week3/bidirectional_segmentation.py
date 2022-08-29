import time


# 读取词表
def load_prefix_word_dict(path):
    prefix_dict = {}                # 创建空词典存储词表
    with open(path, encoding='utf-8') as f:
        for line in f:
            word = line.split()[0]      # 取词
            prefix_dict[word] = 1       # 真词赋1
            for i in range(1, len(word)):       # 对词的前缀切片，北 => 北京、北京大 => 北京大学 ······
                if word[:i] not in prefix_dict:
                    prefix_dict[word[:i]] = 0   # 如果切片后的前缀不在prefix_dict里，赋0
    return prefix_dict


# 实现正向最大匹配
def forward_segmentation(string, prefix_dict):
    if string == '':
        return []
    words = []
    start_index, end_index = 0, 1
    window = string[start_index: end_index]
    find_word = window
    while start_index < len(string):
        if window not in prefix_dict or end_index > len(string):
            words.append(find_word)
            start_index += len(find_word)
            end_index = start_index + 1
            window = string[start_index:end_index]
            find_word = window

        elif prefix_dict[window] == 0:
            end_index += 1
            window = string[start_index: end_index]

        elif prefix_dict[window] == 1:
            find_word = window
            end_index += 1
            window = string[start_index: end_index]
    return words


# 实现逆向最大匹配
def backward_segment(string, prefix_dict):
    list = []
    i = len(string) - 1
    while i >= 0:
        long_word = string[i]
        for j in range(0, i):
            word = string[j:i + 1]
            if word in prefix_dict:
                if len(word) > len(long_word):
                    long_word = word
                    break
        list.append(long_word)
        i -= len(long_word)
    return list


# 计算单字数量
def number_words(list):
    # print(sum(1 for word in list if len(word) == 1))
    return sum(1 for word in list if len(word) == 1)


# 同时执行正向和逆向最长匹配，若两者的词数不同，则返回词数更少的一个,否则返回两者中单字更少的那一个。当单字也相同时，优先返回正向最长匹配结果
def bidirectional_segmentation(string, prefix_dict):
    b = backward_segment(string, prefix_dict)
    f = forward_segmentation(string, prefix_dict)
    if len(f) > len(b):
        return b
    elif len(f) < len(b):
        return f
    else:
        if number_words(f) > number_words(b):
            return b
        else:
            return f


def main(bidirectional_segmentation, input_path, out_path):
    prefix_dict = load_prefix_word_dict('dict.txt')
    writer = open(out_path, 'w', encoding='utf-8')
    start_time = time.time()
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            words = bidirectional_segmentation(line.strip(), prefix_dict)
            # print('/'.join(words) + '\n')
            writer.write('/'.join(words) + '\n')
    writer.close()
    print('耗时', time.time() - start_time)
    return


if __name__ == '__main__':
    prefix_dict = load_prefix_word_dict('dict.txt')
    # string = "北京大学生前来报到"
    string = "王羲之草书《平安帖》共有九行"
    # string = "非常的幸运"
    # main(bidirectional_segmentation, 'corpus.txt', 'bidirectional_method_output.txt')
    print(forward_segmentation(string, prefix_dict), len(forward_segmentation(string, prefix_dict)))
    print(backward_segment(string, prefix_dict), len(backward_segment(string, prefix_dict)))
    print(bidirectional_segmentation(string, prefix_dict))
