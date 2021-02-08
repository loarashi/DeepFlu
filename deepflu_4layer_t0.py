# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:24:56 2020

@author: anna
"""

import pandas as pd
from sklearn import preprocessing
import sys

import all_22277_list

all_df = pd.read_csv("data/t0_data/H1N1_rmat0_001_t0_no237/H1N1_rmat0_001_t0_no237.txt",sep='\t',encoding='utf-8')#讀取H1N1_rmat0_001_t0_no237檔案

cols=all_22277_list.cols#將Probe匯入
all_df=all_df[cols]

def PreprocessData(raw_df):#將資料格式化
   
    df=raw_df.drop(['ID'], axis=1)#去除ID
    ndarray = df.values
    Features = ndarray[:,1:]#提取資料特徵
    Label = ndarray[:,0]#提取資料標籤

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))#將特徵minmax
    scaledFeatures=minmax_scale.fit_transform(Features)    
    
    return scaledFeatures,Label

all_Features,all_Label=PreprocessData(all_df)
test_Features=all_Features[:2]#提取測試資料的特徵(兩個時間點)
test_Label=all_Label[:2]#提取測試資料的標籤(兩個時間點)
train_Features=all_Features[2:]#提取訓練資料的特徵
train_Label=all_Label[2:]#提取訓練資料的標籤

from keras.models import Sequential
from keras.layers import Dense,Dropout

model = Sequential()#模型一層輸入層22277個節點，四層隱藏層100個節點，一層dropout參數0.1，一層輸出層1個節點
model.add(Dense(units=100, input_dim=22277, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
train_history =model.fit(x=train_Features, y=train_Label, validation_split=0.1, epochs=150, batch_size=200,verbose=0)
scores=model.evaluate(x=test_Features, y=test_Label)

all_Features,Label=PreprocessData(all_df)
all_probability=model.predict_classes(all_Features)
all_probability_score=model.predict_proba(all_Features)
pd=all_df
pd.insert(len(all_df.columns), 'probability',all_probability)
predict=all_probability[:2]#提取測試資料的預測標籤
predict_score=all_probability_score[:2]#提取測試資料的預測機率
predict=predict.tolist()
predict_score=predict_score.tolist()

f1 = open("./H1N1_t0_001.txt", 'a', encoding = 'UTF-8')#可自行取名儲存檔案
f1.write(str(int(test_Label[0]))+"\t"+str(predict[0])+"\t"+str(predict[1])+"\t"+str(predict_score[0])+"\t"+str(predict_score[1])+"\n")#列印測試資料標籤/測試資料預測標籤(兩個時間點)/測試資料預測機率(兩個時間點)
f1.close
