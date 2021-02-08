# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:24:56 2020

@author: anna
"""

import numpy
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn import preprocessing, metrics

import all_22277_list

all_df = pd.read_csv("data/alltime_data/H1N1_alltime_no237.txt",sep='\t',encoding='utf-8')#讀取H1N1_alltime_no237.txt檔案

cols=all_22277_list.cols#將Probe匯入
all_df=all_df[cols]
msk = numpy.random.rand(len(all_df)) < 0.8#使用8&2隨機分類為訓練資料與測試資料

def PreprocessData(raw_df):#將資料格式化
   
    df=raw_df.drop(['ID'], axis=1)#去除ID
    ndarray = df.values
    Features = ndarray[:,1:]#提取資料特徵
    Label = ndarray[:,0]#提取資料標籤

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))#將特徵minmax
    scaledFeatures=minmax_scale.fit_transform(Features)
    
    return scaledFeatures,Label

all_Features,all_Label=PreprocessData(all_df)
test_Features=all_Features[~msk]#提取測試資料的特徵
test_Label=all_Label[~msk]#提取測試資料的標籤
train_Features=all_Features[msk]#提取訓練資料的特徵
train_Label=all_Label[msk]#提取訓練資料的標籤

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

all_Features,Label=PreprocessData(all_df)
all_probability=model.predict_classes(all_Features)
all_probability_score=model.predict_proba(all_Features)
pd=all_df
pd.insert(len(all_df.columns), 'probability',all_probability)
predict=all_probability[~msk]#提取測試資料的預測標籤
predict_score=all_probability_score[~msk]#提取測試資料的預測機率
predict_score=predict_score.tolist()

fpr, tpr, thresholds = metrics.roc_curve(test_Label, predict)
auc_roc = metrics.auc(fpr, tpr)#計算出AUROC

precision, recall, thresholds = precision_recall_curve(test_Label, predict)
auc_pr = metrics.auc(recall, precision)#計算出AUPR

TP=0
TN=0
FP=0
FN=0
for j in range(predict.size):
    if(predict[j]==1 and predict[j]==test_Label[j]):
        TP=TP+1
    else:
        TP=TP+0
    
    if(predict[j]==0 and predict[j]==test_Label[j]):
        TN=TN+1
    else:
        TN=TN+0
    
    if(test_Label[j]==0 and predict[j]==1):
        FP=FP+1
    else:
        FP=FP+0
    
    if(test_Label[j]==1 and predict[j]==0):
        FN=FN+1
    else:
        FN=FN+0

acc=(TP+TN)/(TP+TN+FP+FN)#計算出Accuracy
sen=TP/(TP+FN)#計算出Sensitivity
spe=TN/(TN+FP)#計算出Specificity
pre=TP/(TP+FP)#計算出Precision
mcc=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) ** 0.5)#計算出MCC
    
print(str(acc)+"\t"+str(sen)+"\t"+str(spe)+"\t"+str(pre)+"\t"+str(mcc)+"\t"+str(auc_roc)+"\t"+str(auc_pr)+"\n")#列印出各指標數值
