# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:24:56 2020

@author: anna
"""

import numpy
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn import cross_validation, ensemble, preprocessing, metrics
import sys

import all_22277_list

all_df = pd.read_csv("data/改237改rma/H1N1_tall_data/finish_run_h1n1_237.txt",sep='\t',encoding='utf-8')

cols=all_22277_list.cols

all_df=all_df[cols]
msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]

def PreprocessData(raw_df):
   
    df=raw_df.drop(['ID'], axis=1)#移除name欄位
    ndarray = df.values#dataframe轉換為array
    Features = ndarray[:,1:] 
    Label = ndarray[:,0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)    
    
    return scaledFeatures,Label

all_Features,all_Label=PreprocessData(all_df)
test_Features=all_Features[~msk]
test_Label=all_Label[~msk]
train_Features=all_Features[msk]
train_Label=all_Label[msk]

from keras.models import Sequential
from keras.layers import Dense,Dropout

model = Sequential()

model.add(Dense(units=100, input_dim=22277,                 kernel_initializer='uniform',                 activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(units=100,                kernel_initializer='uniform',                activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=100,                kernel_initializer='uniform',                 activation='relu'))

model.add(Dense(units=100,                 kernel_initializer='uniform',                 activation='relu'))

model.add(Dense(units=1,                 kernel_initializer='uniform', 
                activation='sigmoid'))
#model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
train_history =model.fit(x=train_Features, 
                         y=train_Label, 
                         validation_split=0.1, 
                         epochs=150, 
                         batch_size=200,verbose=0)

all_Features,Label=PreprocessData(all_df)
all_probability=model.predict_classes(all_Features)
all_probability_score=model.predict_proba(all_Features)
pd=all_df
pd.insert(len(all_df.columns),
          'probability',all_probability)
predict=all_probability[~msk]
predict_score=all_probability_score[~msk]
#predict=predict.tolist()
predict_score=predict_score.tolist()
#predict=all_probability[~msk]
scores=model.evaluate(x=test_Features,
                      y=test_Label)
print(predict)
#roc
fpr, tpr, thresholds = metrics.roc_curve(test_Label, predict)
auc_roc = metrics.auc(fpr, tpr)
#pr

precision, recall, thresholds = precision_recall_curve(test_Label, predict)
auc_pr = metrics.auc(recall, precision)

#print("auc(pr)",auc_pr)
#print("auc(roc)",auc_roc)

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
'''
print(str(TP)+"\t"+str(TN)+"\t"+str(FP)+"\t"+str(FN)+"\t"+str(auc_roc)+"\t"+str(auc_pr)+"\n")
acc=(TP+TN)/(TP+TN+FP+FN)
sen=TP/(TP+FN)
spe=TN/(TN+FP)
pre=TP/(TP+FP)
mcc=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) ** 0.5)
print(acc, sen, spe, pre, mcc)
'''
#f1 = open("./MLP_0608_tall_all_4dd111101/"+sys.argv[2]+".txt", 'a', encoding = 'UTF-8')     
print(str(TP)+"\t"+str(TN)+"\t"+str(FP)+"\t"+str(FN)+"\t"+str(auc_roc)+"\t"+str(auc_pr)+"\n")
