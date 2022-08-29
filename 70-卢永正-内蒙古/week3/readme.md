第一步：产生分词不同的点，即寻找分词不同的点，并保存

```python
for i in range(len(sentence)):
        DAG = []
        n = min(4+i, len(sentence)+1)
        for j in range(i, n):
            if sentence[i:j] in Dict:
                DAG.append(sentence[i:j])
        DAG_list.append(DAG)
```

产生第一个分词：

```python
target = []
    for i in range(len(DAG_list[0])):  # 第一个字 组成词列表中
        len0 = len(DAG_list[0][i])
        new = []
        if len0 >= len(DAG_list):
            new.append([DAG_list[0][i]])
        else:
            for j in range(len(DAG_list[len0])):  # 词组长度对应的 下个字 组成词列表中
                print('i:', i)
                print('j:', i)
                print('DAG_list[0][i]:', DAG_list[0][i])
                print('DAG_list[len0][j]:', DAG_list[len0][j])
                new.append([DAG_list[0][i]]+[DAG_list[len0][j]])
        target.extend(new)
```

根据分词不同的点进行分词

```python
def creat_last(target):  # 调用DAG，以上一个list为基础生成下个list
    new_list = []
    for i in range(len(target)):
        s = ''
        for j in range(len(target[i])):
            s += target[i][j]
        len2 = len(s)
        new = []
        if len2 >= len(DAG_list):
            new.append(target[i])
        else:
            for k in range(len(DAG_list[len2])):  # 词组长度对应的 下个字 组成词列表中
                new.append(target[i]+[DAG_list[len2][k]])
        new_list.extend(new)
    return new_list
```

