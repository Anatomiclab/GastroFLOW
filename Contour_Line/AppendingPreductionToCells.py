#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 00:10:54 2020

@author: chunyinwong
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard
from datetime import date
import time
start = time.process_time()
import os



file=[]
from os import walk
path=r'./Contour_Line/ExtractedData/'
for (dirpath, dirnames, filenames) in walk(path):
    for f in filenames:
        if f[-3:] == 'txt':
            file.append(f)
ref_data = pd.read_csv("./trainingdata.csv")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
ref_X = ref_data.iloc[:, 1:42].values
ref_X=sc.fit_transform(ref_X)

path_model=r'./model_file//'
for root,_,name in walk(path_model):
    model0 = keras.models.load_model(root+name[0])
    model1 = keras.models.load_model(root+name[1])
    model2 = keras.models.load_model(root+name[2])
    model3 = keras.models.load_model(root+name[3])
    model4 = keras.models.load_model(root+name[4])
    model5 = keras.models.load_model(root+name[5])
    model6 = keras.models.load_model(root+name[6])
    model7 = keras.models.load_model(root+name[7])
    model8 = keras.models.load_model(root+name[8])
    model9 = keras.models.load_model(root+name[9])
    model10 = keras.models.load_model(root+name[10])

for f in file:
    filename=f
    
    name='./PredictedData/'+filename[:-4]+'.csv'
    dataname=path+filename
    
    data = pd.read_csv(dataname, sep='\t',encoding="ISO-8859–1")
    data.info()
    
    X = data.iloc[:, 7:48].values
    Id = data.iloc[:, 0].values
    X = sc.transform(X)
    
    Y_pre0=model0.predict(X)
    Y_pre1=model1.predict(X)
    Y_pre2=model2.predict(X)
    Y_pre3=model3.predict(X)
    Y_pre4=model4.predict(X)
    Y_pre5=model5.predict(X)
    Y_pre6=model6.predict(X)
    Y_pre7=model7.predict(X)
    Y_pre8=model8.predict(X)
    Y_pre9=model9.predict(X)
    Y_pre10=model10.predict(X)
        
    data['0']=Y_pre0
    data['1']=Y_pre1
    data['2']=Y_pre2
    data['3']=Y_pre3
    data['4']=Y_pre4
    data['5']=Y_pre5
    data['6']=Y_pre6
    data['7']=Y_pre7
    data['8']=Y_pre8
    data['9']=Y_pre9
    data['10']=Y_pre10

    for index, row in data.iterrows():
        avg=float(sum(row[-11:]))/(11)
        data.loc[index,'prediction']=avg
    
    data=data.drop(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    data.to_csv(name, index=False, float_format='%f')
    
print("Done")

