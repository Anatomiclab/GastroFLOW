import pandas as pd
import numpy as np

csv_data=pd.read_csv(r"Internal/K-SVM.csv")
print(csv_data)
pred=np.asarray(csv_data['id'])
pred_diag=np.asarray(csv_data['pred diag'])
#pred_score=np.asarray(csv_data['Score'])
pred_cut=pred.copy()

for i in range(pred.shape[0]):
    pred_cut[i]=""
    for j in range(len(pred[i].split("-"))-1):
      pred_cut[i]=pred_cut[i]+'-'+pred[i].split("-")[j]
    pred_cut[i]=pred_cut[i][1:]

csv_data=pd.read_csv(r"801010/final_training_gt.csv")
gt=np.asarray(csv_data['id'])
gt_diag=np.asarray(csv_data['diagnosis'])


count=0


f=[]
f_properties=["GT: ","Full Pred: ","Pred Diagnosis: ","GT Diagnosis: ", "TP", "FP", "TN", "FN"]

for j in range(gt.shape[0]):
    f.append([])
    for i in range(pred_cut.shape[0]):
        print(pred_cut[i])
        print(gt[j])
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
            #f[j].append(str(pred_score[i]))
            f[j].append(str(pred_diag[i]))
            f[j].append(str(int(gt_diag[j]=="C")))
            if pred_diag[i]==1 and 1==int(gt_diag[j]=="C"):#TP
                f[j].append(str(1))
            else:
                f[j].append(str(0))
            if pred_diag[i]==1 and 1!=int(gt_diag[j]=="C"):#FP
                f[j].append(str(1))
            else:
                f[j].append(str(0))
            if pred_diag[i]==0 and 1== int(gt_diag[j] == "N"):  # TN
                f[j].append(str(1))
            else:
                f[j].append(str(0))
            if pred_diag[i]==0 and 1 != int(gt_diag[j] == "N"):  # FN
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
final.to_csv(r"internal_result/K-SVM.csv")

