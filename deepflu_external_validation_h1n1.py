# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 12:45:13 2020

@author: anna
"""

import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn import preprocessing, metrics

import all_12023_list

all_df = pd.read_csv("data/external_validation_data/73072_H1N1_external_validation_D3_D4.txt",sep='\t',encoding='utf-8')

cols=all_12023_list.cols
all_df=all_df[cols]

def PreprocessData(raw_df):
   
    df=raw_df.drop(['ID'], axis=1)
    ndarray = df.values
    Features = ndarray[:,1:] 
    Label = ndarray[:,0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)    
    
    return scaledFeatures,Label

all_Features,all_Label=PreprocessData(all_df)
train_Features=all_Features[:40]
train_Label=all_Label[:40]
test_Features=all_Features[38:]
test_Label=all_Label[38:]

from keras.models import Sequential
from keras.layers import Dense,Dropout

model = Sequential()

model.add(Dense(units=100, input_dim=12023, kernel_initializer='uniform', activation='relu'))
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
predict=all_probability[38:]
predict_score=all_probability_score[38:]
predict_score=predict_score.tolist()

fpr, tpr, thresholds = metrics.roc_curve(test_Label, predict)
auc_roc = metrics.auc(fpr, tpr)

precision, recall, thresholds = precision_recall_curve(test_Label, predict)
auc_pr = metrics.auc(recall, precision)

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

acc=(TP+TN)/(TP+TN+FP+FN)
sen=TP/(TP+FN)
spe=TN/(TN+FP)
pre=TP/(TP+FP)
mcc=((TP*TN)-(FP*FN))/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) ** 0.5)
    
print(str(acc)+"\t"+str(sen)+"\t"+str(spe)+"\t"+str(pre)+"\t"+str(mcc)+"\t"+str(auc_roc)+"\t"+str(auc_pr)+"\n")
