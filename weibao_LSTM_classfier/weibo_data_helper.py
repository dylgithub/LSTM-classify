#encoding=utf-8
import pandas as pd
import jieba
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from gensim.models import word2vec
def get_len():
    df = pd.read_csv('data/new_use_data.csv', encoding='utf-8')
    content_list=list(df['content'])
    max_len=0
    for content in content_list:
        if len(content)>max_len:
            max_len=len(content)
    print(max_len)
def get_train_data():
    # fadongji=pd.read_excel('data/fadongji.xlsx')
    # guzhang=pd.read_excel('data/guzhang.xlsx')
    # qita=pd.read_excel('data/qita.xlsx')
    # fadongji['flag']=0
    # guzhang['flag']=1
    # qita['flag']=2
    # need_df=pd.concat([fadongji[1:20001],guzhang[1:20001],qita[1:20001]])
    # need_df=pd.DataFrame(need_df)
    # need_df.reset_index(drop=True,inplace=True)
    # # print(need_df.head(5))
    # ran=np.random.permutation(60000)
    # shuchu=need_df.ix[ran]
    # shuchu=pd.DataFrame(shuchu)
    # df = pd.read_csv('data/use_data.csv', encoding='utf-8')
    # content_list=df['content']
    # new_content_list=[]
    # for content in content_list:
    #     content=content.replace('*','')
    #     content=content.replace('¤','')
    #     content=content.replace('-','')
    #     content=content.replace('>','')
    #     content=content.replace('<','')
    #     content=content.replace('R&R','')
    #     content=content.replace('/','')
    #     content=content.replace('■','')
    #     content=content.replace('【','')
    #     content=content.replace('】','')
    #     new_content_list.append(content)
    # shuchu=pd.DataFrame({
    #     'content':new_content_list,
    #     'flag':list(df['flag'])
    # })
    # shuchu.to_csv('data/new_use_data.csv',index=False)
    jieba.load_userdict('dict/my_dict.txt')
    df = pd.read_csv('data/new_use_data.csv', encoding='utf-8')
    content_list = list(df['content'])
    jieba_cut_list=[jieba.lcut(content) for content in content_list]
    one_hot=OneHotEncoder()
    level=list(df['flag'])
    one_hot_label=one_hot.fit_transform(np.array(level).reshape([-1,1]))
    one_hot_label=one_hot_label.toarray()
    print('final get_train_data')
    return jieba_cut_list,one_hot_label
def train_word2vec_model():
    jieba_cut_list,_=get_train_data()
    word2vec_model = word2vec.Word2Vec(jieba_cut_list,min_count=1, window=2, size=80)
    word2vec_model.save('word2vec_model/weibao_model')
#获得所有单词的词向量
def get_vec_data(sentence_length,vector_size):
    word_cut_list, label=get_train_data()
    model=word2vec.Word2Vec.load('word2vec_model/weibao_model')
    vec_list=[]
    for sentens in word_cut_list:
        word_list = []
        for i,word in enumerate(sentens):
            if i>=sentence_length:
                break
            else:
                word_list.append(model[word])
        #对长度不足的句子进行补零操作
        sen_len = len(sentens)
        if sen_len < sentence_length:
            num = sentence_length - sen_len
            for j in range(num):
                word_list.append([0] * vector_size)
        vec_list.append(np.array(word_list))
    print('final get_vec_data')
    return vec_list,label
# if __name__ == '__main__':
#     train_word2vec_model()
#     get_len()