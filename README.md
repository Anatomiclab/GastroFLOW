# Prioritization on whole slide images of clinical gastric carcinoma biopsies through a weakly supervised and annotation-free system

This repository provides training and testing scripts for the article "Prioritization on whole slide images of clinical gastric carcinoma biopsies through a weakly supervised and annotation-free system".

## Gastric Classification Network

### Data for Cross-Validation, External Validation

* Cross-Validation Data:
`https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EYlJePFwtM1GpSknK0adq18BDO7zwOF63QHHfGkmQqa9Xw`

* External Validation Data:
`https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EQFpIguoZMpHgaCPkJpEEokBTdYUON7_JTXRQa046HFEKQ`

* Retrospctive:
`https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EYtw2btVoQNOgcdiw3gCu4oBOayIsqlA6Ek0gQzljDWotA`


### Network Training

To train a model, use script `ModelTraining.py`.

Script outputs:

* **(TPR, TNR, PPV, NPV, Accuracy, roc_auc, val_accuracy).h5**: *.h5* file which is the model checkpoint after training.

To find the optimized parameters of model using Talos, use script `talos_tunning.py`

Script outputs:

* **(tuningtime).csv**: *.csv* file which contains the finding parameters and the corresponding loss value and validation metrics.

### Cross-Validation

In order to validate the performance of integrated model, use script `Validation_internal.py`

Modified Line(please fill the path after downloading the Internal Data.):

Line 26: `ref_data = pd.read_csv("801010/train_cross10.csv")`

Line 44: `data = pd.read_csv("801010/test_cross10.csv")`

Modified Line(please fill the path for saving the results.):

Line 162,163: `report_pos2.to_csv("Network_result/" + posName)`,`report_neg2.to_csv("Network_result/" + negName)`




