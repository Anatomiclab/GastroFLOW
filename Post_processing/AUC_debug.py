import glob
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

filename1=r"C:\Users\admin\PycharmProjects\Feature_Flow_Gitver\ForAUC\stage2_result\EX1_2_stage2_test_PN_0.2.csv"
result_csv1 = pd.read_csv(filename1)
gt = list(result_csv1['GT Diagnosis: '])
score1 = list(result_csv1['Pred Score: '])

gt_numpy = np.asarray(gt)
score_numpy1 = np.asarray(score1)

filename2=r"C:\Users\admin\PycharmProjects\Feature_Flow_Gitver\ForAUC\stage2_result\EX1_2_stage2_test_PN_0.3.csv"
result_csv2 = pd.read_csv(filename2)
gt = list(result_csv2['GT Diagnosis: '])
score2 = list(result_csv2['Pred Score: '])

gt_numpy = np.asarray(gt)
score_numpy2 = np.asarray(score2)

#print(score_numpy1.shape)
#print(score_numpy2.shape)

for i in range(score_numpy1.shape[0]):
   if score_numpy1[i]!=score_numpy2[i]:
      print(i)