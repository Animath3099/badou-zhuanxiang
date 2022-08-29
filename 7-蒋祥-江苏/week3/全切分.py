import jieba

def calc_dag(sentence):
        DAG = {}    #DAG空字典，用来存储DAG有向无环图
        N = len(sentence)
        for k in range(N):
            tmplist = []
            i = k
            frag = sentence[k]
            while i < N:
                if frag in Dict:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

#将DAG中的信息解码（还原）出来，用文本展示出所有切分方式
class DAGDecode:
    #通过两个队列来实现
    def __init__(self, sentence):
        self.sentence = sentence
        self.DAG = calc_dag(sentence)
        self.length = len(sentence)
        self.unfinish_path = [[]]
        self.finish_path = []

    def decode_next(self, path):
        path_length = len("".join(path))
        if path_length == self.length:
            self.finish_path.append(path)
            return
        candidates = self.DAG[path_length]
        new_paths = []
        for candidate in candidates:
            new_paths.append(path + [self.sentence[path_length:candidate+1]])
        self.unfinish_path += new_paths
        return

    def decode(self):
        while self.unfinish_path != []:
            path = self.unfinish_path.pop()
            self.decode_next(path)

if __name__ == "__main__":
    # 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
    Dict = {"经常": 0.1,
            "经": 0.05,
            "有": 0.1,
            "常": 0.001,
            "有意见": 0.1,
            "歧": 0.001,
            "意见": 0.2,
            "分歧": 0.2,
            "见": 0.05,
            "意": 0.05,
            "见分歧": 0.05,
            "分": 0.1}

    # 待切分文本
    sentence = "经常有意见分歧"

    # 目标输出;顺序不重要
    target = [
        ['经常', '有意见', '分歧'],
        ['经常', '有意见', '分', '歧'],
        ['经常', '有', '意见', '分歧'],
        ['经常', '有', '意见', '分', '歧'],
        ['经常', '有', '意', '见分歧'],
        ['经常', '有', '意', '见', '分歧'],
        ['经常', '有', '意', '见', '分', '歧'],
        ['经', '常', '有意见', '分歧'],
        ['经', '常', '有意见', '分', '歧'],
        ['经', '常', '有', '意见', '分歧'],
        ['经', '常', '有', '意见', '分', '歧'],
        ['经', '常', '有', '意', '见分歧'],
        ['经', '常', '有', '意', '见', '分歧'],
        ['经', '常', '有', '意', '见', '分', '歧']
    ]
    dag = DAGDecode(sentence)
    dag.decode()
    print(dag.finish_path)