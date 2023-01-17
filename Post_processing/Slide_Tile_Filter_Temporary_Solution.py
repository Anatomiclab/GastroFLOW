import pandas as pd
import numpy as np
import glob
import time
start_time=time.time()

stage1_csv=pd.read_csv(r"Network/External_requireTile_.csv")
stage1_id=stage1_csv['ID']

stage1_csv_pos=pd.read_csv(r"Network/External_report_pos_.csv")
stage1_id_pos=stage1_csv_pos['ID']

final_list = []


csv_properties=["id","Avg Score","positiveNum","diagnosis"]


threshold=0.2
filename=r"Final_Tile_Stage2/External_500_"+str(threshold)+".csv"

stage2_csv=pd.read_csv(filename)
# Diagnosis
print(stage2_csv)
final_list = []
count = 0


for i in range(stage1_id_pos.shape[0]):
        final_list.append([])
        final_list[count].append(stage1_csv_pos['ID'][i].split("_")[0])
        final_list[count].append(stage1_csv_pos['Score'][i])
        final_list[count].append(str(0))
        final_list[count].append(stage1_csv_pos['predict'][i])
        count+=1


stage2_id=stage2_csv['id']

for i in range(stage1_id.shape[0]):
    for j in range(stage2_id.shape[0]):
        if stage1_id[i].split("_")[0]==stage2_id[j]:
                final_list.append([])

                final_list[count].append(stage2_csv['id'][j])
                final_list[count].append(stage2_csv['Avg Score'][j])
                final_list[count].append(stage2_csv['positiveNum'][j])
                final_list[count].append(stage2_csv['diagnosis'][j])
                count += 1

case_final = pd.DataFrame(columns=csv_properties, data=final_list)
case_final.to_csv(r"Final_Tile_Stage1_2/"+filename.split('/')[-1],index=False)

end_time=time.time()
print("Total Time: ",end_time-start_time)