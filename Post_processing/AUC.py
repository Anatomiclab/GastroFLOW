
import glob
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

for filename in glob.glob(r"ForAUC\\*\\*"):
   result_csv = pd.read_csv(filename)
   print(filename)
   gt = list(result_csv['GT Diagnosis: '])
   score = list(result_csv['Pred Score: '])

   gt_numpy = np.asarray(gt)
   score_numpy = np.asarray(score)

   fpr, tpr, thresholds = metrics.roc_curve(gt_numpy, score_numpy, pos_label=1)
   roc_auc = metrics.auc(fpr, tpr)
   print(roc_auc)