import os

import pandas as pd
import numpy as np
import glob
import time


threshold=0.2


final_list=[]
count=0


res_time=time.time()

for filename in glob.glob(r"case_split_Sorted/*"):
    final_list.append([])

    positive_counter=0
    avg_score=0

    csv_data = pd.read_csv(filename)
    # Diagnosis
    #print(csv_data)


    for i in range(csv_data.shape[0]):
        if int(csv_data["Positive Num"][i])>5:
            positive_counter+=1

    for i in range(int(threshold*csv_data.shape[0])):
            avg_score+=float(csv_data["Avg Score"][i])

    if positive_counter>int(threshold*csv_data.shape[0]):
        final_list[count].append(csv_data['id'][5].split("_")[0])
        final_list[count].append(str(float(avg_score/(int(threshold*csv_data.shape[0])))))
        #final_list[count].append(str(float(avg_score)))
        final_list[count].append(str((int(threshold * csv_data.shape[0]))))
        final_list[count].append(str(positive_counter))
        final_list[count].append("Suspect positive")
    elif csv_data.shape[0]>5:
        final_list[count].append(csv_data['id'][5].split("_")[0])
        final_list[count].append(str(float(avg_score / (int(threshold*csv_data.shape[0])))))
        #final_list[count].append(str(float(avg_score)))
        final_list[count].append(str((int(threshold * csv_data.shape[0]))))
        final_list[count].append(str(positive_counter))
        final_list[count].append("negative")

    count+=1

os.makedirs("Final_Tile_Stage2//", exist_ok=True)

csv_properties=["id","Avg Score","Passline","positiveNum","diagnosis"]
case_final = pd.DataFrame(columns=csv_properties, data=final_list)
case_final.to_csv(r"Final_Tile_Stage2//EX1_2_" + str(siz) +"_"+str(threshold)+ ".csv", index=False)


print(time.time()-res_time)

