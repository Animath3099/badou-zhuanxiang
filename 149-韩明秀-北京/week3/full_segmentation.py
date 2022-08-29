import os,sys
import copy
prefix_dict = dict()
all_seg_res = []
seg_res = []
#倒入词表
#值为0代表是前缀
#值为1代表是一个词且这个词向后没有更长的词
#值为2代表是一个词，但是有比他更长的词
def loadWords(word_file):
    global prefix_dict
    with open(word_file,"r") as f:
        for line in f:
            word = line.strip().split(":")[0]
            for i in range(1, len(word)):
                if word[:i] not in prefix_dict: #不能用前缀覆盖词
                    prefix_dict[word[:i]]=0  #前缀
                if prefix_dict[word[:i]] == 1:
                    prefix_dict[word[:i]] = 2
            if word not in prefix_dict:
                prefix_dict[word] = 1  #词
            else:
                if prefix_dict[word] == 0:
                    prefix_dict[word] = 2

def isInWordDict(word,start_idx,end_idx):
    if(end_idx<=len(word)):
        tmpWord = word[start_idx:end_idx]
        if tmpWord in prefix_dict:
            return True
        else:
            return False
    else:
        return False

def find_segmentation(word,start_idx):
    global all_seg_res
    global seg_res
    if start_idx == len(word):
        all_seg_res.append(copy.deepcopy(seg_res))
        return
    end_idx = start_idx + 1
    while end_idx <= len(word):        
        if isInWordDict(word,start_idx,end_idx):
            if prefix_dict[word[start_idx:end_idx]] == 0:
                end_idx += 1
            elif prefix_dict[word[start_idx:end_idx]] == 2:
                seg_res.append(word[start_idx:end_idx])
                find_segmentation(word,end_idx)
                if len(seg_res)>=1:
                    seg_res.pop()
                end_idx += 1
            else:
                seg_res.append(word[start_idx:end_idx])
                find_segmentation(word,end_idx)
                if len(seg_res)>=1:
                    seg_res.pop()
                break
        else:
            seg_res.append(word[start_idx:end_idx])
            find_segmentation(word,end_idx)
            if len(seg_res)>=1:
                seg_res.pop()
            break
def main(word,words_file):
    global all_seg_res
    if len(word) == 0 or len(words_file) == 0:
        return [[word]]
    else:
        loadWords(words_file)
        if len(prefix_dict) == 0:
            return [[word]]
        start_idx = 0
        find_segmentation(word,start_idx)
        for result in all_seg_res:
            print(result)
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("python work3.py test_string words_file")
    else:
        main(sys.argv[1],sys.argv[2])
