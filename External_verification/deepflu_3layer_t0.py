# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:24:56 2020

@author: anna
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn import cross_validation, ensemble, preprocessing, metrics
import sys

import all_12023_list

all_df = pd.read_csv("external_verification_data/73072_H1N1_303_allt0_ta5te7_ID.txt",sep='\t',encoding='utf-8')

cols=all_12023_list.cols

all_df=all_df[cols]

def PreprocessData(raw_df):
   
    df=raw_df.drop(['ID'], axis=1)#移除name欄位
    ndarray = df.values#dataframe轉換為array
    Features = ndarray[:,1:] 
    Label = ndarray[:,0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)    
    
    return scaledFeatures,Label

all_Features,all_Label=PreprocessData(all_df)
test_Features=all_Features[:2]
test_Label=all_Label[:2]
train_Features=all_Features[2:]
train_Label=all_Label[2:]

from keras.models import Sequential
from keras.layers import Dense,Dropout

model = Sequential()
model.add(Dense(units=100, input_dim=12023, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
model.summary()

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.

opts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=opts))

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
train_history =model.fit(x=train_Features, 
                         y=train_Label, 
                         validation_split=0.1, 
                         epochs=150, 
                         batch_size=200,verbose=0)
scores=model.evaluate(x=test_Features,
                      y=test_Label)

all_Features,Label=PreprocessData(all_df)
all_probability=model.predict_classes(all_Features)
all_probability_score=model.predict_proba(all_Features)
pd=all_df
pd.insert(len(all_df.columns), 'probability', all_probability)
predict=all_probability[:2]
predict_score=all_probability_score[:2]
predict=predict.tolist()
predict_score=predict_score.tolist()

f1 = open("./MLP_0924_4d1111b01_52428_H1N1_all_A/"+sys.argv[2]+".txt", 'a', encoding = 'UTF-8')     
f1.write(str(int(test_Label[0]))+"\t"+str(predict[0])+"\t"+str(predict[1])+"\t"+str(predict_score[0])+"\t"+str(predict_score[1])+"\n")
f1.close
