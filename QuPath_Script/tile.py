import os
import pandas as pd
import numpy as np





tile_path = "./Training_tile/RAW_TXT-SET_20200520/"
ratio = os.listdir(tile_path)

for r in ratio:
    df = pd.DataFrame()
    for tile in os.listdir(tile_path+r):
        df_tmp = pd.read_csv("{}{}/{}".format(tile_path,r,tile))
        df = pd.concat([df, df_tmp])

    df = df.drop(['X','Y','Centroid X µm','Centroid Y µm', 'cell_count'],axis = 1)
    print(df.columns)
    df.to_csv("./RAW_TXT-SET_20200520_tiledata_{}.csv".format(r), index=False)

