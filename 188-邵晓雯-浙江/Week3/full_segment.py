
input_path = 'corpus.txt'
output_path = 'fullcut.txt'
writer = open(output_path,'w',encoding='utf-8')
def load_pre_dict(path):
    pre_dict = {}
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            word = line.split()[0].strip()
            for i in range (len(word)):
                if word[:i+1] not in pre_dict:
                    pre_dict[word[:i+1]] = 0
            pre_dict[word[:i+1]] = 1
    return  pre_dict

def dfs(start_index,add_words,text,pre_dict):
    if start_index >= len(text):
        print(add_words)
        writer.write('/'.join(add_words)+'\n')
        return

    end_index = start_index
    for i in range(start_index,len(text)):
        word = text[start_index:i+1]
        if word in pre_dict:
            if pre_dict[word] == 1:
                end_index = i
                add_words.append(word)
                dfs(i+1,add_words,text,pre_dict)
                add_words.pop()
            else:

                if end_index+1 == len(text):
                    add_words.append(word[start_index:start_index+1])
                    dfs((start_index+1,add_words,text,pre_dict))
                    add_words.pop()
                    i = end_index + 1

        else:
            i = end_index + 1

def main(cutmethod,input_path,output_path):
    add_words = []
    pre_dict = load_pre_dict(path)
    with open(input_path,'r',encoding='utf-8')as f:
        for line in f:
            print(line)
            dfs(0,add_words,line,pre_dict)
    writer.close()

path ='dict.txt'
pre_dict = load_pre_dict(path)
add_words = []
# string = '我是中国人'
#string = "分析师指出"
#string = '我是北京大学学生'
#string = '美国哥伦比亚大学国际工商研究院主任魏尚进和国际粮食政策研究所资深研究员张晓波一同进行的最新研究显示'
#dfs(0,add_words,string,pre_dict)
main(dfs,input_path,output_path)
