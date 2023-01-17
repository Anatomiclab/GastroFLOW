import os
import shutil

import pandas as pd
import numpy as np
import glob
import time

res_time=time.time()

folder_path=r"avg_score/"

csv_data=pd.read_csv(folder_path+r"External_tiledata_500.csv")



gt_data=csv_data.copy()
gt_id=gt_data['id'].copy()


for i in range(len(gt_data['id'])):
  null_str=''
  for j in range(len(gt_data['id'][i].split('-')[0:-1])):
      null_str=null_str+'_'+gt_data['id'][i].split('-')[j]
  gt_id[i]=null_str[1:]


print(len(gt_id))

case_id=[]

case_count=-1

temp_id=gt_id[0]

for i in range(len(gt_id)):
    if i==0:
        case_id.append([])
        case_count += 1
        case_id[case_count].append(gt_id[i])
        temp_id = gt_id[i]
    elif temp_id!=gt_id[i]:
        case_id.append([])
        case_count+=1
        case_id[case_count].append(gt_id[i])
        temp_id=gt_id[i]
    elif temp_id==gt_id[i]:
        case_id[case_count].append(gt_id[i])


print(len(case_id))

csv_properties=gt_data.columns
print(csv_properties)

double_counter=0

for i in range(len(case_id)):
    double_counter+=len(case_id[i])

print(double_counter)

case_data=[]
case_data.append([])

mark_label=0
current_cursor=0

gt_data_numpy=np.asarray(gt_data)

for i in range(len(gt_id)):
    if gt_id[i]==gt_id[current_cursor]:
        case_data[int(mark_label)].append(gt_data_numpy[i])
    else:
        mark_label+=1
        current_cursor=i
        case_data.append([])
        case_data[int(mark_label)].append(gt_data_numpy[i])

double_counter=0

shutil.rmtree(r"case_split//")
os.makedirs(r"case_split//",exist_ok=True)

for i in range(len(case_data)):
  case_final=pd.DataFrame(columns=csv_properties,data=case_data[i])
  case_final.to_csv(r"case_split//"+str(i)+".csv",index=False)

print(time.time()-res_time)