# -*- coding: utf-8 -*-

# !/usr/bin/env python
# coding: utf-8


import tensorflow as tf
from tensorflow import keras
# import keras
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard
from datetime import date
from os import walk
import time
import math

thershold = 0.5  # Tested from 0.1 to 0.5 with 0.1 intervals

print("######Running Start###############")

stage1_1_start = time.time()
print("Stage1_1 Start time: ", stage1_1_start)

ref_data = pd.read_csv("./trainingdata.csv")

print(tf.config.list_physical_devices('GPU'))
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
ref_X = ref_data.iloc[:, 1:42].values
Id = ref_data.iloc[:, 0].values
ref_X = sc.fit_transform(ref_X)
np.save("X_train_norm.npy",ref_X)
np.save("X_train_id.npy",Id)

stage1_1_end = time.time()
print("Stage1_1 end time: ", stage1_1_end)
print("Stage1_1 time: ", stage1_1_end - stage1_1_start)


def read(model_path):
    model = keras.models.load_model(model_path)

    data = pd.read_csv("data/External_SlideData.csv")

    X = data.iloc[:, 1:42].values
    Id = data.iloc[:, 0].values

    X = sc.transform(X)


    Y_pre = model.predict(X).reshape(len(data.index), )

    result = pd.DataFrame({'id': Id, 'pre_probi': Y_pre})
    columns = ['ID', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'Score', 'PredictDiagnosis']
    report = pd.DataFrame(columns=columns)

    for index, row in result.iterrows():
        newString = ''
        if '-' in row[0]:
            newString = row[0][:row[0].rfind('-')]
        elif '.' in row[0]:
            newString = row[0][:row[0].rfind('.')]

        result.loc[index, 'id'] = newString
    ID = result['id']

    for c in ID.unique():
        s = result.loc[result['id'] == c, 'pre_probi']

        pre = all(i >= .5 for i in list(s))

        series = [c] + list(s)

        if len(series) != 10:
            for count in range(10 - len(series)):
                series = series + ['NA']
        if len(series) == 10:
            series = series + [float(max(s))] + [pre]
            ap = pd.Series(series, index=columns)
        report = report.append(ap, ignore_index=True)
    return (report)



path = r'model_file/'

stage1_2_start = time.time()
print("Stage1_2 Start time: ", stage1_2_start)

result = pd.DataFrame()
for root, _, name in walk(path):
    for f in name:
        report = read(root + f)
        if result.shape == (0, 0):
            result = report.loc[:, ['ID', 'Score']]
            result.columns = ['ID', f]
        else:
            result[f] = report.loc[:, 'Score']


stage1_2_end = time.time()
print("Stage1_2 end time: ", stage1_2_end)
print("Stage1_2 time: ", stage1_2_end - stage1_2_start)

stage1_3_start = time.time()
print("Stage1_3 Start time: ", stage1_3_start)

result['predict'] = 'Error'
result['Score'] = 10000000
for index, row in result.iterrows():
    List = list(row[1:-2])
    # print(len(List))
    booleanList = []
    for ele in List:
        if ele >= .5:
            booleanList = booleanList + [True]
        else:
            booleanList = booleanList + [False]
    count = sum(booleanList * 1)

    avg = float(sum(row[1:-2])) / (len(row[1:-2]))
    result.loc[index, 'Score'] = avg

    if count >= 6:
        result.loc[index, 'predict'] = 'Positive'
    elif count < 6:
        result.loc[index, 'predict'] = 'False'
    else:
        result.loc[index, 'predict'] = 'Error'



result_stage1=result.copy()

stage1_3_end = time.time()
print("Stage1_3 end time: ", stage1_3_end)
print("Stage1_3 time: ", stage1_3_end - stage1_3_start)

stage1_4_start = time.time()
print("Stage1_4 Start time: ", stage1_4_start)

report_neg = result.loc[result['predict'] == 'False']
report_pos = result.loc[result['predict'] == 'Positive']


posName = 'External_report_pos_' +  '.csv'
negName = 'External_requireTile_' + '.csv'



report_pos2 = report_pos.copy()
report_pos2.sort_values(by='Score', ascending=False, inplace=True)
report_neg2 = report_neg.copy()
report_neg2.sort_values(by='Score', ascending=False, inplace=True)

report_pos.sort_values(by='Score', ascending=False)
report_neg.sort_values(by='Score', ascending=False)




report_pos2.to_csv("Network_result/" + posName)
report_neg2.to_csv("Network_result/" + negName)

stage1_4_end = time.time()
print("Stage1_4 end time: ", stage1_4_end)
print("Stage1_4 time: ", stage1_4_end - stage1_4_start)

