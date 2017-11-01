# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:13:31 2017
FASTTEXT
@author: yaohongfu
"""

#数据预处理
import jieba
import pandas as pd
import fasttext
def get_seg(strs):
    seg_list = jieba.cut(strs,cut_all=False)
    s=str()
    for i in seg_list:
        s=s+' '+i
    return s

text=pd.read_csv(r"E:\yhf\ddf\t.txt",sep='\t',header=None)

text.columns=['id','title','content','lable']
text['seg_title']=text['title'].apply(get_seg)
text['seg_content']=text['content'].apply(get_seg)


head=r'__label__'
num=len(text)
File = open("hello.txt","w")
for i in range(num):
    line=head+text.iloc[i]['lable']+' '+' '+text.iloc[i]['seg_title']+' '+' '+text.iloc[i]['seg_content']
    print(line)
    File.write(line+"\n")
File.close()



lr=0.1
epoch=100
dim=200
bucket=0
model_name='model_ai'
inputf='hello.txt'
wev='word2vec.v2.bin'
classifier = fasttext.supervised(input_file=inputf,output=model_name,epoch=epoch,dim=dim,bucket=bucket,pretrained_vectors=wev)


result = classifier.test('test.txt')
print('P@1:', result.precision)
print('R@1:', result.recall)
print('Number of examples:', result.nexamples)

