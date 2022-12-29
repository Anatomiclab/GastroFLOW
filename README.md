# Prioritization on whole slide images of clinical gastric carcinoma biopsies through a weakly supervised and annotation-free system

This repository provides training and testing scripts for the article "Prioritization on whole slide images of clinical gastric carcinoma biopsies through a weakly supervised and annotation-free system".

## Gastric Classification Network

### Data for Cross-Validation, External Validation

* Cross-Validation Data:

[Internal Data](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EYlJePFwtM1GpSknK0adq18BDO7zwOF63QHHfGkmQqa9Xw)

* External Validation Data:

[External Data](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EQFpIguoZMpHgaCPkJpEEokBTdYUON7_JTXRQa046HFEKQ)

* Retrospctive:

[Retrospective Set](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EYtw2btVoQNOgcdiw3gCu4oBOayIsqlA6Ek0gQzljDWotA)


### Network Training

To train a model, use script `ModelTraining.py`.

**Script outputs**:

* **(TPR, TNR, PPV, NPV, Accuracy, roc_auc, val_accuracy).h5**: *.h5* file which is the model checkpoint after training.

To find the optimized parameters of model using Talos, use script `talos_tunning.py`

**Script outputs**:

* **(tuningtime).csv**: *.csv* file which contains the finding parameters and the corresponding loss value and validation metrics.

### Cross-Validation

In order to validate the performance of integrated model in Cross-Validation, use script `Validation_internal.py`

**Modified Line(please fill the path after downloading the Internal Data.)**:

Line 26: `ref_data = pd.read_csv("801010/train_cross10.csv")`

Line 44: `data = pd.read_csv("801010/test_cross10.csv")`

**Modified Line(please fill the path for saving the results.)**:

Line 162,163: `report_pos2.to_csv("Network_result/" + posName)`,`report_neg2.to_csv("Network_result/" + negName)`

**Script outputs**:

* **report_pos_.csv**: *.csv* file which contains the Slides model predict positive.

* **requireTile_.csv**: *.csv* file which contains the Slides model predict negative.

### External Validation

#### Network Prediction

In order to validate the performance of integrated model in External Validation, it divides into three parts:

* **Only Slide Data**

Use script: `Validation_stage1.py`

**Modified Line(please fill the path after downloading the External Data.)**:

Line 27: `ref_data = pd.read_csv("./trainingdata.csv")`

Line 47: `data = pd.read_csv("EX1_EX2_SlideData.csv")`

**Modified Line(please fill the path for saving the results.)**:

Line 166,167: `report_pos2.to_csv("Network_result/" + posName)`,`report_neg2.to_csv("Network_result/" + negName)`

**Script outputs**:

* **report_pos_.csv**: *.csv* file which contains the Slides model predict positive.

* **requireTile_.csv**: *.csv* file which contains the Slides model predict negative.

After generate the predicted result, to generate the triage list of result, please follow the section **Post Processing**.

* **Only Tile Data**

Use script: `Validation_stage2.py`

**Modified Line(please fill the path after downloading the External Data.)**:

Line 27: `ref_data = pd.read_csv("./trainingdata.csv")`

Line 44: `data = pd.read_csv("ex1_ex2_tiledata_500.csv")`

**Modified Line(please fill the path for saving the results.)**:

Line 69: `table.to_csv("Network_result/"+ "EX1_EX2_tiledata_500_nocut.csv")`

**Script outputs**:

* **"EX1_EX2_tiledata_500_nocut.csv"**: *.csv* file which contains the predicted result of Tiled Data.

After generate the predicted result, to generate the triage list of result, please follow the section **Post Processing**.

* **Slide and Tile Data**

Use script: `Validation_stage1_2.py`

**Modified Line(please fill the path after downloading the External Data.)**:

Line 27: `ref_data = pd.read_csv("./trainingdata.csv")`

Line 44: `data = pd.read_csv("ex1_ex2_tiledata_500.csv")`

**Modified Line(please fill the path for saving the results.)**:

Line 69: `table.to_csv("Network_result/"+ "EX1_EX2_tiledata_500_nocut.csv")`

**Script outputs**:

* **report_pos_.csv**: *.csv* file which contains the Slides model predict positive.

* **requireTile_.csv**: *.csv* file which contains the Slides model predict negative.

* **"EX1_EX2_tiledata_500.csv"**: *.csv* file which contains the predicted result of Tiled Data for the slides model predict negative.

 
#### Post-Processing for generating Triage List

After predicting the slides and tiled data, to generate the triage list for analysis, please follow the steps below:

* **Cross-Validation**

Use script: `Post_processing\Slide_internal_Stage1_generation.py`

**Modified Line(please fill the path after running the prediction.)**:

Line 4: `csv_data=pd.read_csv(r"Network_result/Internal_cross10_report_pos_.csv"))`

Line 14: `csv_data=pd.read_csv(r"801010/final_training_gt.csv")`

Line 64: `data = pd.read_csv("EX1_EX2_SlideData.csv")`

**Modified Line(please fill the path for saving the results.)**:

Line 166,167: `report_pos2.to_csv("Network_result/" + posName)`,`report_neg2.to_csv("Network_result/" + negName)`

**Script outputs**:

* **report_pos_.csv**: *.csv* file which contains the Slides model predict positive.

* **requireTile_.csv**: *.csv* file which contains the Slides model predict negative.





