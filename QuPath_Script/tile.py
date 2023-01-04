import os
import pandas as pd
import numpy as np





ex1 = "./Training_tile/RAW_TXT-SET_20200520/"
ratio = os.listdir(ex1)

for r in ratio:
    df = pd.DataFrame()
    for tile in os.listdir(ex1+r):
        df_tmp = pd.read_csv("{}{}/{}".format(ex1,r,tile))
        df = pd.concat([df, df_tmp])

    df = df.drop(['X','Y','Centroid X µm','Centroid Y µm', 'cell_count'],axis = 1)
    print(df.columns)
    df.to_csv("./RAW_TXT-SET_20200520_tiledata_{}.csv".format(r), index=False)

