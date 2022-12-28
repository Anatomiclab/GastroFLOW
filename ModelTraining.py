# -*- coding: utf-8 -*-

from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K 
from sklearn.metrics import confusion_matrix, roc_curve, auc
import math

import numbers
import os
import six

import numpy
import matplotlib.collections
from matplotlib import pyplot
from sklearn import metrics
from sklearn.metrics import auc

name='Model'


def run(X, Y, Id_n, data, input_model=None):
    random.shuffle(Id_n, random.random)
    Id_test=Id_n[:int(len(Id_n)*.1)]
    Id_val=Id_n[int(len(Id_n)*.1):int(len(Id_n)*.2)]
    Id_train=Id_n[int(len(Id_n)*.2):]
    
    
    test_df=data.iloc[0:0, :]
    
    train_df=data.iloc[0:0, :]
    
    val_df=data.iloc[0:0, :]
    
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
    
    for a in Id_val:
        s=a+'-1'
        li=[s]
        temp=data.loc[data.id.isin(li)]
        val_df=val_df.append(temp, ignore_index=True)
    
        s=a+'-2'
        li=[s]
        temp=data.loc[data.id.isin(li)]
        val_df=val_df.append(temp, ignore_index=True)
    
        s=a+'-3'
        li=[s]
        temp=data.loc[data.id.isin(li)]
        val_df=val_df.append(temp, ignore_index=True)
    
        
        
    test_df = test_df.reindex(np.random.permutation(data.index)) 
    train_df = train_df.reindex(np.random.permutation(data.index))
    val_df = val_df.reindex(np.random.permutation(data.index))
    
    test_df=test_df.dropna()
    train_df=train_df.dropna()
    val_df=val_df.dropna()
    
    X_train=train_df.iloc[:, 1:42].values
    Y_train=train_df.iloc[:, 42].values
    
    X_val=val_df.iloc[:, 1:42].values
    Y_val=val_df.iloc[:, 42].values
    
    X_test=test_df.iloc[:, 1:42].values
    Y_test=test_df.iloc[:, 42].values    
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_test=sc.transform(X_test)
    X_train=sc.transform(X_train)
    X_val=sc.transform(X_val)
    
    model = keras.Sequential()
    model.add(layers.Dense(95,input_dim=41, activation = "relu",activity_regularizer=regularizers.l2(0.35)))
    model.add(layers.Dropout(0.05))
    model.add(layers.Dense(95, activation = "relu"))
    model.add(layers.Dropout(0.15))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer ='Nadam',loss='logcosh', metrics =['accuracy'])
    

    
    X_test = K.cast_to_floatx(X_test)
    Y_test = K.cast_to_floatx(Y_test)
    X = K.cast_to_floatx(X)
    Y = K.cast_to_floatx(Y)
    X_train1 = K.cast_to_floatx(X_train)
    Y_train1 = K.cast_to_floatx(Y_train)
    X_val = K.cast_to_floatx(X_val)
    Y_val = K.cast_to_floatx(Y_val)

    history=model.fit(X_train1, Y_train1, batch_size=42, epochs=20, validation_data=(X_val, Y_val))
    
    Y_pred_num=model.predict(X_test)
    Y_pred =(Y_pred_num>0.5)
    
    cm = confusion_matrix(Y_test, Y_pred)
    
    conf_matrix=cm
    TN = conf_matrix[0][0]  # True negatives
    TP = conf_matrix[1][1]  # True positives
    FN = conf_matrix[1][0]  # False negatives
    FP = conf_matrix[0][1]  # False positives
    
    TPR = float(TP)/(TP+FN)
    print('TPR (a.k.a. recall) = %4.2f%%' % (TPR*100))
    
    TNR = float(TN)/(TN+FP)
    print('TNR = %4.2f%%' % (TNR*100))
    
    PPV = float(TP)/(TP+FP)
    print('PPV = %4.2f%%' % (PPV*100))
    
    NPV = float(TN)/(TN+FN)
    print('NPV = %4.2f%%' % (NPV*100))
    
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    print('Accuracy = %4.2f%%' % (Accuracy*100))
    
    Val_accuracy=history.history['val_accuracy'][-1]
    print(f'Validation Accuracy = {Val_accuracy}')
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred_num)
    roc_auc = auc(fpr, tpr),2    
    analysis=[TPR, TNR, PPV, NPV, Accuracy, roc_auc, history.history['val_accuracy'][-1]]
    
    myStr=', '.join(map(str,analysis))
    global i
    if Accuracy >.85 and history.history['val_accuracy'][-1] > .85:
        i+=1
        try:
            os.mkdir(f'./{name}/')
        except:
            print('file exist')


        model.save(f'./{name}/{myStr}.h5')
    
    Y_pre_num=model.predict(X)
    Y_pre =(Y_pre_num>0.5)
    
    Y_pre=model.predict(X).reshape(len(X),)
    result = pd.DataFrame({'pre_probi':Y_pre, 'res':Y})
    result['pre']=(result['pre_probi']>0.5)*1
    result['corr']=(result['pre']==result['res'])
    result['id']=Id
    result['corr']= ~result['corr']
    result=result[['id', 'res','pre_probi','corr']]

    indexNames = result[result['corr'] == False].index
    result = result.drop(indexNames)
    result=result[['id', 'res','pre_probi']]
    del model
    return(result, analysis)


data = pd.read_csv("2.2 Model Building/trainingdata.csv")
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

import random


columns = ['id', 'deviation','number of error', 'average']
result_B = pd.DataFrame(columns=['run','TPR', 'TNR', 'PPV', 'NPV', 'Accuracy', 'AUC', 'Val_Accu'])
i=0

for a in range(750):
    

    result_df, analysis_list=run(X, Y, Id_n, data)
    if i>10:
        break
    
    no=[a+1]
    analysis_list=no+analysis_list
    analysis_list=pd.Series(analysis_list, index=['run','TPR', 'TNR', 'PPV', 'NPV', 'Accuracy','AUC', 'Val_Accu'])
    result_B=result_B.append(analysis_list,ignore_index=True)
    print(f'Number of cycle: {a+1}')


result_B.to_csv(f"./{name}/SummaryOfPerformance.csv", index=False)
    