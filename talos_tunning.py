#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K 
from tensorflow.keras import metrics 
import random
import tensorflow
import talos as ta
from talos.model.early_stopper import early_stopper
import pickle
import time
from talos.utils import hidden_layers
print(tensorflow.__version__)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def model(x_train, y_train, x_val, y_val, params):
    model = keras.Sequential()
    model.add(layers.Dense(params['first_neuron'],input_dim=41, activation = params["activation"],activity_regularizer=regularizers.l2(params['L2'])))
    model.add(layers.Dropout(params['do']))

    hidden_layers(model, params, 1)
        
    model.add(layers.Dense(1, activation=params['last_activation']))
    model.compile(loss=params['losses'],
              optimizer=params['optimizer'],
              metrics=['accuracy', metrics.binary_accuracy,f1_m])

    history = model.fit(x_train, y_train, 
                        validation_data=(x_val, y_val),
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0,
                        callbacks=early_stopper(params['epochs'], mode='strict'))
    return history, model

p={
   'first_neuron': (5, 105, 10),
   'activation':['relu', 'elu'],
   'L2':[0.05,0.15,0.25,0.35],
   'do':[0.05,0.15,0.25,0.35],
   'hidden_layers':[0,1,2],
   'last_activation':['sigmoid'],
   'optimizer':['Adam', 'Nadam', 'RMSprop'],
   'losses': ['logcosh', 'binary_crossentropy'],
   'epochs': (10, 81, 10),
   'batch_size': (10, 50, 10),
   'shapes':['brick', 'triangle', 'funnel'],
   'dropout': [0.05,0.15,0.25,0.35],
       }

#read and clean data
data = pd.read_csv("trainingdata.csv")
data = data.drop_duplicates(keep='first')
data = data.reindex(np.random.permutation(data.index))
data.info()

count = 0
for c in data.iloc[:,42]:
    if c == "C":
        data.iloc[count,42] = 1
        count+=1
    elif c == "N":
        data.iloc[count,42] = 0
        count+=1

data.head(5)

#divide data
Id=data.iloc[:, 0].values
X = data.iloc[:, 1:42].values
Y = data.iloc[:, 42].values
data.describe(include='all')

Id=data['id']

Id=list(Id)
Id_n=[]
for s in Id:
    s=s[:s.rfind('-')]
    Id_n.append(s)
Id_n=list(set(Id_n))  

random.shuffle(Id_n, random.random)
Id_test=Id_n[:int(len(Id_n)*.1)]
Id_train=Id_n[int(len(Id_n)*.1):]


test_df=data.iloc[0:0, :]
train_df=data.iloc[0:0, :]

for a in Id_test:
    s=a+'-1'
    li=[s]
    temp=data.loc[data.id.isin(li)]
    test_df=test_df.append(temp, ignore_index=True)

    s=a+'-2'
    li=[s]
    temp=data.loc[data.id.isin(li)]
    test_df=test_df.append(temp, ignore_index=True)

    s=a+'-3'
    li=[s]
    temp=data.loc[data.id.isin(li)]
    test_df=test_df.append(temp, ignore_index=True)

for a in Id_train:
    s=a+'-1'
    li=[s]
    temp=data.loc[data.id.isin(li)]
    train_df=train_df.append(temp, ignore_index=True)

    s=a+'-2'
    li=[s]
    temp=data.loc[data.id.isin(li)]
    train_df=train_df.append(temp, ignore_index=True)

    s=a+'-3'
    li=[s]
    temp=data.loc[data.id.isin(li)]
    train_df=train_df.append(temp, ignore_index=True)

    
test_df = test_df.reindex(np.random.permutation(data.index)) 
train_df = train_df.reindex(np.random.permutation(data.index))

test_df=test_df.dropna()
train_df=train_df.dropna()

X_train=train_df.iloc[:, 1:42].values
Y_train=train_df.iloc[:, 42].values


X_test=test_df.iloc[:, 1:42].values
Y_test=test_df.iloc[:, 42].values    

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test=sc.transform(X_test)
X_train=sc.transform(X_train)

X_test = K.cast_to_floatx(X_test)
Y_test = K.cast_to_floatx(Y_test)
X = K.cast_to_floatx(X)
Y = K.cast_to_floatx(Y)
X_train = K.cast_to_floatx(X_train)
Y_train = K.cast_to_floatx(Y_train)
    
print('starting........')
t=ta.Scan(x=X_train, y=Y_train, model=model, params=p, x_val=X_test, y_val=Y_test, fraction_limit=.005, experiment_name='hyperparameter_tuning',
          random_method='quantum')

with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(t, f)


