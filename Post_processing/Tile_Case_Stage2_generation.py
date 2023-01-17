import pandas as pd
import numpy as np
import time

start_time=time.time()

threshold=0.2

csv_data=pd.read_csv(r"Final_Tile_Stage2/External_500_"+str(threshold)+".csv")
print(csv_data)
pred=np.asarray(csv_data['id'])
pred_diag=np.asarray(csv_data['diagnosis'])
pred_score=np.asarray(csv_data['Avg Score'])
pred_cut=pred.copy()

for i in range(pred.shape[0]):
    pred_cut[i]=pred[i].split("_")[0]

csv_data=pd.read_csv(r"data/GroundTruth_External.csv")
gt=np.asarray(csv_data['ID'])
gt_diag=np.asarray(csv_data['Diagnosis'])

count=0


f=[]
f_properties=["GT: ","Full Pred: ","Pred Score: ","Pred Diagnosis: ","GT Diagnosis: ", "TP", "FP", "TN", "FN"]

for j in range(gt.shape[0]):
    f.append([])
    for i in range(pred_cut.shape[0]):
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
            f[j].append(str(int(pred_diag[i]=='Suspect positive')))
            f[j].append(str(int(gt_diag[j]=="CANCER")))
            if int(pred_diag[i]=='Suspect positive')==1 and int(gt_diag[j]=="CANCER")==1:#TP
                f[j].append(str(1))
                f[j].append(str(0))
                f[j].append(str(0))
                f[j].append(str(0))
            elif int(pred_diag[i]=='Suspect positive')==1 and int(gt_diag[j]=="CANCER")==0:#FP
                f[j].append(str(0))
                f[j].append(str(1))
                f[j].append(str(0))
                f[j].append(str(0))
            elif int(pred_diag[i]=='Suspect positive')==0 and int(gt_diag[j]=="CANCER")==0:#TN
                f[j].append(str(0))
                f[j].append(str(0))
                f[j].append(str(1))
                f[j].append(str(0))
            elif int(pred_diag[i]=='Suspect positive')==0 and int(gt_diag[j]=="CANCER")==1:#FN
                f[j].append(str(0))
                f[j].append(str(0))
                f[j].append(str(0))
                f[j].append(str(1))


    print("Total Pair: ",count)
print(f)

f2=[]
for i in range(len(f)):
    if len(f[i])>0:
        f2.append(f[i])

final=pd.DataFrame(columns=f_properties,data=f2)
final.to_csv(r"stage2_result/External_stage2_test_PN_"+str(threshold)+".csv")

end_time=time.time()
print("Total Time: ",end_time-start_time)
