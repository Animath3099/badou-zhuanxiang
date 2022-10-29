import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("文本分类练习.csv",header=None)
    trainData = df.sample(frac=0.7)
    df = pd.concat([df,trainData],axis=0,ignore_index=True)
    df.drop_duplicates(keep=False,inplace=True)
    trainData.to_csv("train.csv",header=False,index=False)
    df.to_csv("valid.csv", header=False, index=False)