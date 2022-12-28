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


path = r'model_file/'

stage2_1_start = time.time()
print("Stage2_1 Start time: ", stage2_1_start)


data = pd.read_csv("ex1_ex2_tiledata_500.csv")

X = data.iloc[:, 1:42].values
Id = data.iloc[:, 0].values
stage2_1_end = time.time()
print("Stage2_1 end time: ", stage2_1_end)
print("Stage2_1 time: ", stage2_1_end - stage2_1_start)

X = sc.transform(X)

stage2_2_start = time.time()
print("Stage2_2 Start time: ", stage2_2_start)



table = pd.DataFrame()
for root, _, name in walk(path):
    for f in name:
        model = keras.models.load_model(root + f)
        Y_pre = model.predict(X).reshape(X.shape[0], )

        if table.shape == (0, 0):
            table = pd.DataFrame({'id': Id, f: Y_pre})
        else:
            table[f] = Y_pre
    table.to_csv("Network_result/"+ "EX1_EX2_tiledata_500_nocut.csv")

stage2_2_end = time.time()
print("Stage2_2 end time: ", stage2_2_end)
print("Stage2_2 time: ", stage2_2_end - stage2_2_start)