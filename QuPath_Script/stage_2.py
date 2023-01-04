import os
import numpy as np
import pandas as pd





ex1 = "./2022Gastrointernaldataraw/RAW_TXT-SET_20200520/"
tile_ratio = [500]




sep_file=ex1.split('/')[-2]

saving_path="./Training_tile/"

os.mkdir(saving_path + sep_file)

for file in os.listdir(ex1):

    for ratio in tile_ratio:

        df = pd.read_csv(ex1+file, header=0,sep="	")
        #print(df['Centroid X µm'])
        try:
            df.insert(loc=7, column = "X", value = np.floor(df['Centroid X µm'].values/(ratio/4)))
            df.insert(loc=8, column = "Y", value = np.floor(df['Centroid Y µm'].values/(ratio/4)))
        except:
            print(file)

        df_new = df.groupby(['X','Y']).mean()
        df_new['cell_count'] = df.groupby(['X', 'Y']).size()
        df_new.reset_index(inplace=True)
        df_new.insert(loc=0, column = 'id', value = file[:-4])
        df_new = df_new.loc[df_new.cell_count>10]
        print(df_new)
        
        if os.path.exists(saving_path+sep_file+"/"+"{}".format(ratio))<=0:
            os.mkdir(saving_path+sep_file+"/"+"{}".format(ratio))
        df_new.to_csv(saving_path+sep_file+"/"+"{}/{}_tiledata.csv".format(ratio, file[:-4]), index=False)