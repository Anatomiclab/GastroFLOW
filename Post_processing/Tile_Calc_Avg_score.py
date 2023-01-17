import pandas as pd
import numpy as np
import glob
import time

res_time=time.time()



folder_path=r"Network_result//"

csv_data=pd.read_csv(folder_path+r"External_tiledata_500.csv")
pred=np.asarray(csv_data['id'])
pred_cut=pred.copy()

avg_score_list=[]
positive_num_list=[]
f_properties=["id","Avg Score","Positive Num"]
for i in range(pred_cut.shape[0]):
    avg_score=0
    pos_num=0
    for j in range(2,13):
        score_line=csv_data[csv_data.columns[j]][i]
        avg_score+=float(score_line)
        if float(score_line)>0.5:
            pos_num+=1
    avg_score=float(avg_score)/11

    avg_score_list.append(avg_score)
    positive_num_list.append(pos_num)

f=[]
for i in range(pred_cut.shape[0]):
  if avg_score_list[i]>0:
    f.append([])
    f[i].append(csv_data['id'][i])
    f[i].append(avg_score_list[i])
    f[i].append(positive_num_list[i])


final=pd.DataFrame(columns=f_properties,data=f)
final.to_csv(r"avg_score//"+"External_tiledata_500.csv")


print(time.time()-res_time)
