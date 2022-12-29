import pandas as pd
import numpy as np

import time

start_time=time.time()

csv_data=pd.read_csv(r"Network/EX1_EX2_report_pos_.csv")
print(csv_data)
pred=np.asarray(csv_data['ID'])
pred_diag=np.asarray(csv_data['predict'])
pred_score=np.asarray(csv_data['Score'])
pred_cut=pred.copy()

for i in range(pred.shape[0]):
    pred_cut[i]=pred[i].split("_")[0]

csv_data=pd.read_csv(r"Ground_Truth/GroundTruth_ex1_2_noex_nodup.csv")
gt=np.asarray(csv_data['ID'])
gt_diag=np.asarray(csv_data['Diagnosis'])

count=0

f=[]
f_properties=["GT: ","Full Pred: ","Pred Score: ","Pred Diagnosis: ","GT Diagnosis: ", "TP", "FP", "TN", "FN"]

for j in range(gt.shape[0]):
    f.append([])
    for i in range(pred_cut.shape[0]):
        #print(pred[i])
        #print(gt[j])
        print("Pair: ",i,"  ",j)
        if pred_cut[i].replace("%20"," ")==gt[j] and len(f[j])==0:

            count+=1
            print("Pair Found")
            print("Pred num: ", i+2)
            print("GT num: ",j+2)
            print("Pred: ",pred_cut[i])
            print("GT: ", gt[j])

            f[j].append(gt[j])
            f[j].append(pred[i].replace("%20"," "))
            f[j].append(str(pred_score[i]))
            f[j].append(str(1))
            f[j].append(str(int(gt_diag[j]=="CANCER")))
            if 1==int(gt_diag[j]=="CANCER"):#TP
                f[j].append(str(1))
            else:
                f[j].append(str(0))
            if 1!=int(gt_diag[j]=="CANCER"):#FP
                f[j].append(str(1))
            else:
                f[j].append(str(0))
            f[j].append(str(0))
            f[j].append(str(0))


    print("Total Pair: ",count)

print(f)

#f.remove([])


csv_data=pd.read_csv(r"Network/EX1_EX2_requireTile_.csv")
#Diagnosis
print(csv_data)
pred=np.asarray(csv_data['ID'])
pred_diag=np.asarray(csv_data['predict'])
pred_score=np.asarray(csv_data['Score'])
pred_cut=pred.copy()

for i in range(pred.shape[0]):
    pred_cut[i]=pred[i].split("_")[0]


for j in range(gt.shape[0]):
    f.append([])
    for i in range(pred_cut.shape[0]):
        #print(pred[i])
        #print(gt[j])
        print("Pair: ",i,"  ",j)
        if pred_cut[i].replace("%20"," ")==gt[j] and len(f[j])==0:

            count+=1
            print("Pair Found")
            print("Pred num: ", i+2)
            print("GT num: ",j+2)
            print("Pred: ",pred_cut[i])
            print("GT: ", gt[j])

            f[j].append(gt[j])
            f[j].append(pred[i].replace("%20"," "))
            f[j].append(str(pred_score[i]))
            f[j].append(str(0))
            f[j].append(str(int(gt_diag[j]=="CANCER")))
            f[j].append(str(0))
            f[j].append(str(0))
            if 0==int(gt_diag[j]=="CANCER"):#TN
                f[j].append(str(1))
            else:
                f[j].append(str(0))
            if 0!=int(gt_diag[j]=="CANCER"):#FN
                f[j].append(str(1))
            else:
                f[j].append(str(0))


    print("Total Pair: ",count)

print(f)

f2=[]
for i in range(len(f)):
    if len(f[i])>0:
        f2.append(f[i])

final=pd.DataFrame(columns=f_properties,data=f2)
final.to_csv(r"stage1_result/EX1_EX2_stage1_test_PN.csv")

end_time=time.time()
print("Total Time: ",end_time-start_time)


