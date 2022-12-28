import glob
import os
import shutil

import pandas as pd
import numpy as np
import glob
import time

res_time=time.time()

shutil.rmtree("case_split_Sorted//")
os.makedirs("case_split_Sorted//",exist_ok=True)

for filename in glob.glob(r"case_split/*"):
    print(filename)
    gt_data=pd.read_csv(filename)
    gt_data.sort_values(by='Avg Score',inplace=True,ascending=False)
    gt_data.to_csv(filename.replace("case_split","case_split_Sorted"),index=False)
print(time.time()-res_time)