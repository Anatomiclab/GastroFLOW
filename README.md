# Accelerating gastric carcinoma diagnosis with a weakly supervised artificial intelligence-based system by prioritizing gastric biopsy whole slide images

This repository provides scripts (Gastric Carcinoma classification Network "GCNet" and Gastric-specific Prioritization Workflow "GastroFLOW") for the article "Accelerating gastric carcinoma diagnosis with a weakly supervised artificial intelligence-based system by prioritizing gastric biopsy whole slide images".

GCNet is the Gastric Carcinoma classification Network developed in the study. It is an ensemble of optimized Multilayer Perceptron (MLP) neural networks trained on downsampled cellular features extracted from gastric biopsy whole slide images (WSIs). GCNet is designed to classify WSIs as either carcinoma (CA) or non-carcinoma (non-CA) cases. The top-performing MLP networks, based on Area Under the Curve (AUC) values, are selected and ensembled together. A majority voting scheme is used to classify WSIs based on the sum of votes from the MLP-based networks. The average of the MLP network outputs is used to calculate the mean case malignancy score, which is a measure of the probability of a case being CA. This score is used to prioritize cases for further review.

GastroFLOW, on the other hand, is the Gastric Case Prioritization Workflow developed in the study. It integrates GCNet and additional processing steps to enhance the triage ability for CA cases, especially those with low tumor area content. GastroFLOW involves the tiling of non-CA WSIs into smaller image tiles and re-evaluation using GCNet. Tiled images with a mean malignancy prediction score above a certain threshold are considered positive for carcinoma (P-CA). Cases are then assigned a grading based on the original GCNet prediction and the P-CA prediction from the tiled images. Cases graded as CA or suspicious for carcinoma are prioritized for further review, while those graded as benign are deemed lower risk. GastroFLOW aims to improve the detection and prioritization of CA cases, particularly those with low tumor content.


## Content

[Overview of the datasets presented in this study](#overview-of-the-datasets-presented-in-this-study)

[Data used in this studyn](#data-used-in-this-study)

[Python Environment Setup](#python-environment-setup)

[Export WSI and extract cellular features from WSI](#export-wsi-and-extract-cellular-features-from-wsi)

[WSI Data Generation](#wsi-data-generation)

[Tiled Image Data Generation](#tiled-image-data-generation)

[Cross-Validation and External Validation of Machine Learning Algorithms](#cross-validation-and-external-validation-of-machine-learning-algorithms)

* [Cross-Validation](#cross-validation)

* [External-Validation](#external-validation)

[Hyperparameter Optimization using Talos](#hyperparameter-optimization-using-talos)

[MLP Network Training](#mlp-network-training)

[Cross-Validation of GCNet](#cross-validation-of-gcnet)

[External Validation](#external-validation-1)

* [GCNet with WSI Data](#gcnet-with-wsi-data)

* [GCNet with Tiled Image Data](#gcnet-with-tiled-image-data)

* [GastroFLOW](#gastroflow)

[Post-Processing](#post-processing)

* [Cross-Validation for GCNet](#cross-validation-for-gcnet)

* [External Validation](#external-validation-1)

* [GCNet with WSI Data](#gcnet-with-wsi-data-1)

* [GCNet with Tiled Image Data](#gcnet-with-tiled-image-data)

* [GastroFLOW](#gastroflow)

[Localization of Carcinoma Cells and Generation of Cancer Probability Heatmap](#localization-of-carcinoma-cells-and-generation-of-cancer-probability-heatmap)

* [Classification of carcinoma cells](#classification-of-carcinoma-cells)

* [Generation of Cancer Probability Heatmap](#generation-of-cancer-probability-heatmap)

* [Generation of computer-aided diagnosis with contour for highlighting carcinoma regions](#generation-of-computer-aided-diagnosis-with-contour-for-highlighting-carcinoma-regions)

[Network Running time](#network-running-time)

## Overview of the datasets presented in this study

| Dataset |Cases | Non-carcinoma | Carcinoma | Benign to Malignant ratio | WSIs  |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Internal dataset  | 649 | 407 (62.71%) | 242 (37.29%) | 1.68:1 | 2064 |
| External validation dataset  | 312 | 222 (71.15%) | 90 (28.85%) | 2.46:1 | 739 |
| Retrospective case-control study dataset | 90 |60 (66.67%) | 30(33.33%) | 2:1 | 113 |

## Data used in this study

* Internal Data for model training: [Training Dataset](https://connectpolyu-my.sharepoint.com/:x:/g/personal/21118855r_connect_polyu_hk/EQ16M5yOAvtAiuQGKHwtIagBaHGgOwICKB6DLU8fUc2usQ?e=NmtRhD)

* Cross-Validation Data: [Cross-Validation Data](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EYlJePFwtM1GpSknK0adq18BDO7zwOF63QHHfGkmQqa9Xw)

* External Validation Data: [External Dataset](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EV7M2OkvBV5OlJQCqXtC_vgBlvTacytjn1yn9ptzf5gqfg)

* Retrospective Case-Control Study Dataset: [Retrospective Case-Control Study Dataset](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EYtw2btVoQNOgcdiw3gCu4oBOayIsqlA6Ek0gQzljDWotA)

## Python Environment Setup **(please update the environment,especially the talos)**

To reproduce the Python environment used in the project, please use the environment file is named `environment_tf2.yml` and is based on the Anaconda environment.

By following this step, you will be able to set up the Python environment with the required dependencies and packages used in the project.

## Export WSI and extract cellular features from WSI

To extract the Whole Slide Images (WSI) and cellular features, please follow the steps outlined below:

![Image](Figure_QuPath.png)

Step 1: Export the WSI
1. Open the Software QuPath(0.2.0-m8)[Link](https://qupath.github.io/) and access the script editor by navigating to Automate -> Show script editor.
2. Load the script for image extraction located at `QuPath_Script\ImagesExport.groovy`.
3. Modify Line 7 in the script to specify the path where the extracted images will be saved: 

Line 7: `path="images/"` //**(Specify the path to store the exported images)** 

Script Outputs:

(WSI Images).png: A .png file containing the extracted WSI images

Step 2: Extract the Cellular Features
1. Open the script editor in QuPath software (version 0.2.0-m8).
2. Load the script for cellular feature extraction located at `QuPath_Script\CellularFeaturesExtractionforWholeSlide.groovy.`
3. Modify Line 92 in the script to specify the path where the extracted features will be saved:

Line 92: `save_path = "Feature/"`  //**(Specify the path for saving the extracted features)** 

Script Outputs:

(WSI cellular features).txt: A .txt file containing the extracted cellular features for the WSI images.

By following these steps, you will be able to export the desired WSI images and extract cellular features using the provided QuPath scripts, and exported `(WSI Images).png` and `(WSI cellular features).txt` can be used to generate cancer probability heatmap in the section **"Localization of Carcinoma Cells and Generation of Cancer Probability Heatmap"**.

## WSI Data Generation

Once you have extracted the cellular features, you can generate Whole Slide Image (WSI) data using the provided Python script, `DataAggregationforWholeSlides.py`. Follow the steps below:

Step 1: Run the script `DataAggregationforWholeSlides.py`.

Step 2: Modify Line 7 in the script to specify the path of the cellular features extracted from QuPath:

Line 7: `path=r'./RawData'`//**(Specify the path to the extracted cellular features)** 

Step 3: Modify Line 8 in the script to specify the path where the WSI data will be saved:

Line 8: `outPath=r'./aggregateddata.csv'`//**(Specify the path for saving the WSI data)** 

Script Outputs:

aggregateddata.csv: This is a .csv file that contains the generated WSI data. This file will include relevant information derived from the averaged cellular features of the WSIs.

By following these steps, you will be able to generate the WSI data using the provided Python script. 

## Tiled Image Data Generation 

To generate tiled image data for model training and validation, follow the steps below:

Step 1: Open the script `QuPath_Script\stage_2.py`
1. Open the script `stage_2.py` located in the `QuPath_Script` directory.
2. Modify Line 9 in the script to specify the path where the cellular features of the WSIs generated from QuPath are stored:

Line 9: `feat_path = "./2022Gastrointernaldataraw/RAW_TXT-SET_20200520/"` **(Specify the path for the extracted cellular features)** 

3. Modify Line 17 in the script to specify the immediate path for saving the tile data:

Line 17: `saving_path="./Training_tile/"` **(Specify the immediate path for saving the tile data)**

4. Adjust the tile ratio as needed. By default, the extracted tile size is set to 500x500 pixels.

Line 10: `tile_ratio = [500]` **(Specify the desired tile size)**

Step 2: Run `QuPath_Script\tile.py`
1. Open the script `tile.py` located in the `QuPath_Script` directory.
2. Modify Line 9 in the script to specify the path of the saving path used in Step 1:

Line 9: `tile_path = "./Training_tile/RAW_TXT-SET_20200520/"` **(Fill in the immediate path from Step 1)**

3. Modify Line 20 in the script to specify the path for saving the final tile data:

Line 20: `df.to_csv("./RAW_TXT-SET_20200520_tiledata_{}.csv".format(r), index=False)` **(Fill in the path for saving the final tile data)**

Script Outputs:

(WSI cellular features)_tiledata.csv: This is a .csv file containing the extracted 41 features for the tile image data.

By following these steps, you will be able to generate tiled image data for model training and validation purposes. The resulting tiledata.csv file will contain the extracted cellular features for the tile image data.

## Cross-Validation and External Validation of Machine Learning Algorithms

This section provides the necessary scripts to perform cross-validation on the internal dataset.

### Cross-Validation

To perform cross-validation using machine learning algorithms after downloading cross-validation data, follow these steps below:

Step 1: Run the script `machinelearn_internal.py`

1. Modify the following lines in the script to specify the data paths. Note that the cross-validation fold needs to be modified manually:

Line 111: `Train_data_path="801010/"` **(Specify the root path of the internal data)**

Line 116: `data = pd.read_csv(Train_data_path+'train_cross10.csv')` **(Specify the file name of the training data)**

Line 133: `data2 = pd.read_csv(Train_data_path+'test_cross10.csv')` **(Specify the file name of the testing data)**

2. Modify Line 112 in the script to specify the path for saving the results:

Line 112: `Result_path=""`**(Specify the path for saving the results)**

Script output:

(Machine Learning Algorithm).csv: A .csv file containing the case ID, case WSIs' prediction (CA or non-CA) and the ground truth label(CA or non-CA). **("1" indicates CA, while "0" indicates non-CA)**

Step 2: Run the script `Machine_learning_internal_generation.py`

1. Modify the following lines to specify the data paths. Note that the cross-validation fold needs to be modified manually:

Line 4: `csv_data=pd.read_csv(r"Internal/K-SVM.csv")` **(Specify the path of the predicted results from Step 1)**

Line 17: `csv_data=pd.read_csv(r"801010/final_training_gt.csv")` **(Specify the path of the ground truth)**

2. Modify the following line to specify the path for saving the results:

Line 76: `final.to_csv(r"internal_result/K-SVM.csv")`

Script outputs:

(Machine Learning Algorithm).csv: A .csv file containing the case ID, case WSIs' prediction (CA or non-CA) and classification outcome (True Positive "TP", True Negative "TN", False Positive "FP", False Negative "FN")

By following these steps, you will be able to evaluate cross-validation performance of different machine learning model from its confusion matrix. 

### External Validation 

This section provides instructions for running machine learning algorithms on the external dataset, follow these steps below:

Step 1: Run the script `machinelearn_external_score.py`

1. Modify the following lines in the script to specify the data paths. Note that the cross-validation fold needs to be modified manually:

Line 116: `data = pd.read_csv('trainingdata.csv')` **(Specify the path of the training data)**

Line 135: `data2 = pd.read_csv('data/External_SlideData_gt.csv')` **(Specify the path of the external dataset containing the case WSIs' cellular features with modified (Fake) ground truth.)**

Line 277: `data = pd.read_csv('data/External_SlideData.csv')` **(Specify the path of the external dataset containing the case WSIs' cellular features)**

Line 279: `ref_data = pd.read_csv("data/GroundTruth_External.csv")` **(Specify the path of the case WSIs' ground truth in the external validation dataset)**

2. Modify the following line to specify the path for saving the results:

Line 114: `result_path="external/"`

Script Outputs:

(Machine Learning Algorithm).csv: A .csv file containing the case ID, predicted malignancy prediction scores, and case WSIs' prediction (CA or non-CA) for the external data.**(TRUE indicates CA, while FALSE indicates non-CA)**

Step 2: Run the script `Machine_learning_external_generation.py`

1. Modify the following lines to specify the data paths. Note that the cross-validation fold needs to be modified manually:

Line 4: `csv_data=pd.read_csv(r"SVM--linear/report_pos_.csv")` **(Specify the path of the results for the case WSIs predicted as positive (CA) by the machine learning models)**

Line 13: `csv_data=pd.read_csv(r"data/GroundTruth_External.csv")` **(Specify the path of the case WSI's ground truth in the external validation dataset)**

Line 63: `csv_data=pd.read_csv(r"SVM--linear/requireTile_.csv")` **(Specify the path of the results for the case WSIs predicted as negative (non-CA) by the machine learning models)**

2. Modify the following line to specify the path for saving the results:

Line 115: `final.to_csv(r"external_result/SVM--linear.csv")`

Script Outputs:

(Machine Learning Algorithm).csv: A .csv file containing the case ID, predicted malignancy prediction scores, and case WSIs' prediction (CA or non-CA), ground truth label(CA or non-CA), and classification outcome (True Positive "TP", True Negative "TN", False Positive "FP", False Negative "FN") for the external dataset.

By following these steps, you will be able to evaluate model performance from its confusion matrix, obtain the malignancy prediction scores used for calculating the model's area under the receiver operating curve and generate triage list for cases WSIs. To assess percentage of skipped non-carcinoma cases using different machine learning models, `(Machine Learning Algorithm).csv` can generate triage list using Microsoft Excel by sorting predicted labels (CA and non-CA) first, followed by descending order sorting of malignancy prediction scores.

Note: This section can also be used to obtain malignancy prediction scores used for calculating the model's area under the receiver operating curve and generate triage list for cases WSIs from cross-validation data.

## Hyperparameter Optimization using Talos

To find the optimized parameters for multilayer perceptron (MLP) models using Talos, follow the steps below:

Step 1: Run the script `talos_tunning.py`

1. Modify the following line to specify the path for the training data:

Line 72: `data = pd.read_csv("trainingdata.csv")` **(Specify the path of the training data)**

Script Output:

.csv : A .csv file that contains hyperparameters settings tested along with its performance metrics for training MLP networks. 

By following these steps, you will be able to select the optimized hyperparameters for MLP network training. To train MLP network, please refer to the following section "**MLP Network Training**".

## MLP Network Training

To train any MLP network, follow the steps below using the `ModelTraining.py` script. Ensure that you have set the optimized parameters obtained from Talos to train MLP network.

Step 1: Run the script `ModelTraining.py`

1. Modify the following line in the script to specify the path for the training data:

Line 197: `data = pd.read_csv("trainingdata.csv")` **(Specify the path for the training data)**

Script Outputs:

(TPR, TNR, PPV, NPV, Accuracy, roc_auc, val_accuracy).h5: An .h5 file that serves as the model and its performance after training. This file contains the trained MLP network.

SummaryOfPerformance.csv: A .csv file that provides the performance metrics of the model for different rounds of training.

By following these steps and running the ModelTraining.py script, you will be able to train the MLP networks, and choose suitable MLP networks ensembling as GCNet. The optimized MLP networks used for ensembling as GCNet are stored at `model_file`.


## Cross-Validation of GCNet

This section provides the necessary script, `Validation_internal.py`, to perfrom cross-validation of GCNet.

To cross-validate GCNet, follow these steps:

Step 1: Run the script `Validation_internal.py`

1. Modify the following lines in the script to specify the path after downloading the internal data:

Line 26: `ref_data = pd.read_csv("801010/train_cross10.csv")` **(Specify the file name of the training data)**

Line 44: `data = pd.read_csv("801010/test_cross10.csv")` **(Specify the file name of the testing data)**

2. Modify the following lines to specify the path for saving the results:

Line 146,147: `posName = 'Internal_cross10_report_pos_' +  '.csv'`,`negName = 'Internal_cross10_requireTile_' + '.csv'`

Line 162,163: `report_pos2.to_csv("Network_result/" + posName)`,`report_neg2.to_csv("Network_result/" + negName)`

Script Outputs:

report_pos_.csv: A .csv file containing the cases' WSIs predicted as positive (CA) by the GCNet model. 

requireTile_.csv: A .csv file containing the cases' WSIs predicted as negative (non-CA) by the GCNet model.

Those lists contain the case ID, malignancy prediction scores computed from 11 MLP networks ensembled as GCNet, the case WSIs' prediction **("positive" indicates CA, while "negative" indicate non-CA)**, and the mean malignancy scores of 11 MLP networks ensembled as GCNet.

By following these steps, you will be able to generate the cross-validation result using GCNet. To evaluate the cross-validation performance of GCNet, please refer to the **"Post-Processing"** section.

## External Validation of GCNet and GastroFLOW

The external validation of GCNet and GastroFlow is divided into three parts, each with its respective script.

### GCNet with WSI Data

To validate the performance of GCNet using WSI data from external dataset, follow these steps:

Step 1. Run the script `Validation_stage1.py`

1. Modify the following lines after downloading the external data:

Line 27: `ref_data = pd.read_csv("trainingdata.csv")` **(Specify the file name of the training data)**

Line 47: `data = pd.read_csv("data/External_SlideData.csv")` **(Specify the file name of the testing data)**

2. Modify the following lines to specify the path for saving the results

Line 150,151: `posName = 'External_report_pos_' +  '.csv'`,`negName = 'External_requireTile_' + '.csv'`

Line 166,167: `report_pos2.to_csv("Network_result/" + posName)`,`report_neg2.to_csv("Network_result/" + negName)`

Script Outputs:

report_pos_.csv: A .csv file containing the cases' WSIs predicted as positive (CA) by the GCNet model.

requireTile_.csv: A .csv file containing the cases' WSIs predicted as negative (non-CA) by the GCNet model.

Those lists contain the case ID, malignancy prediction scores computed from 11 MLP networks ensembled as GCNet, the case WSIs' prediction **("positive" indicates CA, while "negative" indicate non-CA)**, and the mean malignancy scores of 11 MLP networks ensembled as GCNet.

### GCNet with Tiled Image Data

To validate the performance of GCNet using tile image data, follow these steps:

Step 1. Run the script `Validation_stage2.py`

1. Modify the following lines after downloading the external data:

Line 27: `ref_data = pd.read_csv("trainingdata.csv")` **(Specify the file name of the training data)**

Line 44: `data = pd.read_csv("data/External_tiledata_500.csv")` **(Specify the file name of the testing data)**

2. Modify Line 69 to specify the path for saving the results:

Line 69: `table.to_csv("Network_result/"+ "External_tiledata_500.csv")`

Script Output:

External_tiledata_500.csv: A .csv file containing the predicted results of tile image data by GCNet.

This list contains the case ID, malignancy prediction scores computed from 11 MLP networks ensembled as GCNet

### GastroFLOW

To validate the performance of GastrolFlow, follow these steps:

Step 1. Run the script `Validation_stage1_2.py`

1. Modify the following lines after downloading the external dataset or retrospective case-control study dataset:

Line 27: `ref_data = pd.read_csv("trainingdata.csv")` **(Specify the file name of the training data)**

Line 47: `data = pd.read_csv("data/External_SlideData.csv")` **(Specify the file name of the WSI testing data)**
 
Line 180: `data = pd.read_csv("data/External_tiledata_500.csv")` **(Specify the file name of the tiled testing data)**

2. Modify the following lines to specify the path for saving the results:

Line 150,151: `posName = 'External_report_pos_' +  '.csv'`,`negName = 'External_requireTile_' + '.csv'`

Line 166,167: `report_pos2.to_csv("Network_result/" + posName)`,`report_neg2.to_csv("Network_result/" + negName)`

Line 230: `table.to_csv("Network_result/"+ "External_tiledata_500.csv")`

Script Outputs:

report_pos_.csv: A .csv file containing the predictions of WSIs as positive (CA).

requireTile_.csv: A .csv file containing the predictions of WSIs as negative (non-CA).

External_tiledata_500.csv: A .csv file containing the predicted results of tile data.

By following these steps, you will be able to generate external validation result using GCNet and GastroFLOW. To evaluate the external validation performance of GCNet and GastroFLOW , please refer to the **"Post-Processing"** section.

 
## Post-Processing

After generating predictions of the WSIs and tiled data using GCNet and GastroFLOW, follow the steps below to evaluate model performance from its confusion matrix, generate the malignancy prediction scores used for calculating the model's area under the receiver operating curve and generate list for case triaging.

### Cross-Validation for GCNet

Step 1. Run the script `Post_processing\Slide_internal_Stage1_generation.py`

1. Modify Line 4 in the script to specify the path of the result for the WSIs predicted as CA

Line 4: `csv_data=pd.read_csv(r"Network_result/Internal_cross10_report_pos_.csv"))` 

2. Modify Line 14 in the script to specify the path of the ground truth label in the internal dataset:

Line 14: `csv_data=pd.read_csv(r"801010/final_training_gt.csv")` 

3. Modify Line 64 in the script to specify the path of the result for the WSIs predicted as non-CA

Line 64: `csv_data=pd.read_csv(r"Network_result/Internal_cross10_requireTile_.csv")` 

4. Modify Line 118 in the script to specify the path for saving the results:

Line 118: `final.to_csv(r"stage1_result/Internal_cross10_test_PN.csv")`

Script outputs:

Internal_.csv: *.csv* file containing predicted results of WSI data by GCNet from cross-validation data. The file includes cases ID, GCNet's predicted scores, predicted labels, ground-truth labels, and classification outcome (True Positive "TP", True Negative "TN", False Positive "FP", False Negative "FN").


### External Validation 

#### GCNet with WSI Data

Step 1. Run the script `Post_processing\Slide_Stage1_generation.py`

1. Modify Line 8 in the script to specify the path of the result for the WSIs predicted as CA

Line 8: `csv_data=pd.read_csv(r"Network/External_report_pos_.csv")` 

2. Modify Line 18 in the script to specify the path of the ground truth in the external validation dataset

Line 18: `csv_data=pd.read_csv(r"data/GroundTruth_External.csv")` 

3. Modify Line 66 in the script to specify the path of the result for the WSIs predicted as non-CA

Line 66: `csv_data=pd.read_csv(r"Network/External_requireTile_.csv")`

4. Modify Line 120 in the script to specify the path for saving the results:

Line 120: `final.to_csv(r"stage1_result/External_stage1_test_PN.csv")`

Script outputs:

`External_stage1_test_PN.csv`: *.csv* file containing predicted results of WSI data by GCNet. The file includes cases ID, GCNet's predicted scores (malignancy prediction scores), predicted daignosis labels **("1" indicates CA, while "0" indicates non-CA)**, ground-truth labels **("1" indicates CA, while "0" indicates non-CA)**, and classification outcome (True Positive "TP", True Negative "TN", False Positive "FP", False Negative "FN").

To generate triage list from `External_stage1_test_PN.csv`, it can use Microsoft Excel by sorting predicted labels (CA and non-CA) first, followed by descending order sorting of malignancy prediction scores.

#### GCNet with Tiled Image Data

Please follow the steps below:

Step 1: Run the script `Post_processing\Tile_Calc_Avg_score.py`

1. Modify Line 10 in the script to specify the folder path after running the network prediction:

Line 10: `folder_path=r"Network_result//"` 

2. Modify Line 12 in the script to specify the filename of the predicted results of tile data by GCNet after running the network prediction:

Line 12: `csv_data=pd.read_csv(folder_path+r"External_tiledata_500.csv")`

3. Modify Line 42 in the script to specify the path for saving the results:

Line 42: `final.to_csv(r"avg_score//"+"External_tiledata_500.csv")` **("avg_score//" is immediate path. Please Create it manually when the path is not existed.)**

Script outputs:

avgscore.csv: *.csv* file containing the predicted result for each tiled image data. The file includes tile image name, GCNet's predicted scores (malignancy prediction scores), and number of MLP networks from GCNet classify tile image is positive to carcinoma (P-CA).

Step 2: Run the script `Post_processing\Tile_Case_split_ver2.py`

1. Modify Line 11 and Line 13 in the script to specify the folder path after running Step 1:

Line 11,13: `folder_path=r"avg_score/"`,`csv_data=pd.read_csv(folder_path+r"External_tiledata_500.csv")` 

2. Modify Line 11 and Line 13 in the script to specify the folder path after running Step 1:

Line 82,83,87: `#shutil.rmtree(r"case_split//")`, `os.makedirs(r"case_split//",exist_ok=True)`, `case_final.to_csv(r"case_split//"+str(i)+".csv",index=False)` **("case_split//" is immediate path. Please create it manually when the path is not existed.)**

Script outputs:

Folder of Case.csv: *.csv* file containing the predicted results of all the tiled image data corresponding to each case.

Step 3: Run the script `Post_processing\Tile_Case_Sort.py`

1. Modify Line 15 in the script to specify the path after running Step 2:

Line 15: `for filename in glob.glob(r"case_split/*"):` **(Fill the immediate path in step 2)**

2. Modify Line 12 and Line 13 in the script to specify the path for saving the results:

Line 12,13: `#shutil.rmtree("case_split_Sorted//")"`,`os.makedirs("case_split_Sorted//",exist_ok=True)` **("case_split_Sorted//" is immediate path. Please create it manually when the path is not existed.)**

Script outputs:

Folder of Case.csv: *.csv* file containing the predicted results of all the tiled image data corresponding to each case. After this step, the scores will be sorted in descending order.

Step 4: Run the script `Post_processing\Tile_Case_Combine_ver1.py`

1. Modify Line 9 in the script to set the cut-off ratio (threshold) of positive for carcinoma (P-CA) tiled images for classifying and prioritizing cases as “suspicious for carcinoma”:

Line 9: `threshold=0.2`

2. Modify Line 18 in the script to specify the path after running Step 3:

Line 18: `for filename in glob.glob(r"case_split_Sorted/*"):` **(Fill the immediate path in step 3)**

3. Modify Line 53 and Line 57 in the script to specify the path for saving the results:

Line 53,57: `os.makedirs("Final_Tile_Stage2//", exist_ok=True)`,`case_final.to_csv(r"Final_Tile_Stage2//External_" +"_"+str(threshold)+ ".csv", index=False)` **("Final_Tile_Stage2//" is immediate path)**

Script outputs:

External_.csv: *.csv* file containing the predicted results for case using external validation dataset. It includes the cases ID, GCNet's predicted scores (malignancy prediction scores) of the top 20% **(tiled images probability threshold)** tile images in each WSI, the minimum number of P-CA tiles required to be classified as suspicious for positive, the actual number of P-CA tiles, and the predicted diagnosis label **("suspective positive" indicates suspicious for carcinoma, while "negative" indicates benign)**.

Step 5: Run the script `Post_processing\Tile_Case_Stage2_generation.py`

1. Modify Line 9 in the script to specify the path after running Step 4:

Line 9: `csv_data=pd.read_csv(r"Final_Tile_Stage2/External_500_"+str(threshold)+".csv")` **(Fill the immediate path in step 4)**

2. Modify Line 19 in the script to specify the path of the ground truth in the external dataset:

Line 19: `csv_data=pd.read_csv(r"data/GroundTruth_External.csv")`  **(Fill the path of ground truth in external dataset)**

3. Modify Line 78 in the script to specify the path for saving the results:

Line 78: `final.to_csv(r"stage2_result/External_stage2_test_PN_"+str(threshold)+".csv")`

Script outputs:

External_stage2_test_PN_.csv: *.csv* file containing the triage list for the external test data based on tiled data. It includes cases ID, GCNet's predicted scores (malignancy prediction scores) of the top 20% **(tiled images probability threshold)** tile images in each WSI, predicted daignosis labels **("1" indicates CA, while "0" indicates non-CA)**, ground-truth labels **("1" indicates CA, while "0" indicates non-CA)**, and classification outcome (True Positive "TP", True Negative "TN", False Positive "FP", False Negative "FN").

By following these steps, you will be able to evaluate model performance from its confusion matrix, obtain the malignancy prediction scores used for calculating the model's area under the receiver operating curve and generate triage list for cases WSIs. To assess percentage of skipped non-carcinoma cases using different machine learning models, `External_stage2_test_PN_.csv` can generate triage list using Microsoft Excel by sorting predicted labels (CA and non-CA) first, followed by descending order sorting of malignancy prediction scores.

#### GastroFLOW

Before proceeding the following steps, make sure to use `External_tiledata_500.csv` generated from `Validation_stage1_2.py` to run the step "GCNet with Tiled Image Data" to generate `External_stage2_test_PN_.csv` file. 

This can ensure only cases predicted as non-CA are reprocessed using their tiled image data, and those case can be reclassified as either suspicious for carcinoma or benign. 

After obtaining `External_stage2_test_PN_.csv` file, please follow the steps:

Step 1: Run the script `Post_processing\Slide_Tile_Filter_Temporary_Solution.py`

1. Modify Line 7 in the script to specify the path of the result for the WSIs predicted non-CA from external dataset or retrospective case-control study dataset:

Line 7: `stage1_csv=pd.read_csv(r"Network/External_requireTile_.csv")` 

2. Modify Line 10 in the script to specify the path of the result for the WSIs predicted CA from external dataset or retrospective case-control study datase:

Line 10: `stage1_csv_pos=pd.read_csv(r"Network/External_report_pos_.csv")`

3. Modify Line 20 in the script to specify the path of the predicted result for the tiled Image Data from external dataset or retrospective case-control study datase:

Line 20: `filename=r"Final_Tile_Stage2/External_500_"+str(threshold)+".csv"` 

4. Modify Line 52 in the script to specify the path for saving the results:

Line 52: `case_final.to_csv(r"Final_Tile_Stage1_2/"+filename.split('/')[-1],index=False)` **("Final_Tile_Stage1_2//" is immediate path)**

Script outputs:

External_.csv: *.csv* file containing the predicted results for the external dataset or retrospective case-control study dataset. It provides the prediction labels (positive, suspect positive, and negative), the predicted scores, and the number of tiled predictions classified as suspect positive.

Step 2: Run the script `Post_processing\Slide_Tile_Stage1_2_generation.py`

1. Modify Line 6 in the script to set the cut-off ratio (threshold) of positive for carcinoma (P-CA) tiled images for classifying and prioritizing cases as “suspicious for carcinoma”:

Line 6: `threshold=0.2` 

2. Modify Line 8 in the script to specify the path after running Step 1:

Line 8: `csv_data=pd.read_csv(r"Final_Tile_Stage1_2/External_500_"+str(threshold)+".csv")` **(Fill the immediate path in step 1)**

3. Modify Line 18 in the script to specify the path of the ground truth in the external dataset or Retrospective Case-Control Study Dataset:

Line 18: `csv_data=pd.read_csv(r"data/GroundTruth_External.csv")` 

4. Modify Line 82 in the script to specify the path for saving the results:

Line 82: `final.to_csv(r"stage1_2_result/External_stage1_2_test_PN_"+str(threshold)+".csv")`

Script outputs:

External_stage1_2_test_PN_.csv: *.csv* file containing the triage list for the external test data based on tiled data. It includes cases ID, GCNet's predicted scores (malignancy prediction scores) of the top 20% **(tiled images probability threshold)** tile images in each WSI, predicted daignosis labels **("1" indicates CA, while "0" indicates non-CA)**, ground-truth labels **("1" indicates CA, while "0" indicates non-CA)**, and classification outcome (True Positive "TP", True Negative "TN", False Positive "FP", False Negative "FN").

By following these steps, you will be able to evaluate model performance from its confusion matrix, obtain the malignancy prediction scores used for calculating the model's area under the receiver operating curve and generate triage list for cases WSIs. To assess percentage of skipped non-carcinoma cases using different machine learning models, `External_stage2_test_PN_.csv` can generate triage list using Microsoft Excel by sorting predicted labels (CA and non-CA) first, followed by descending order sorting of malignancy prediction scores.

## Localization of Carcinoma Cells and Generation of Cancer Probability Heatmap

### Classification of carcinoma cells

Cellular feature and WSI can be generated and exported from section **"Export WSI and extract cellular features from WSI"**

You can generate your own cellular feature and export your WSI data to visualise localization of carcinoma cell and generate cancer probaility heatmap with using our GCNet or your trained MLP network.  

For demostration, sample data for validating the scripts is provided and can be downloaded from the provided [Link](Sample_Feature_For_Generating_Contour_Line.txt), and `Sample_WSI.png`.

![Image](Sample_WSI.png)

Please follow the steps below:

Step 1: Run the script `Contour_Line/AppendingPredictionToCells.py`

1. Modify Line 20 in the script to specify the path of the extracted cellular features:

Line 20: `path=r'./Contour_Line/ExtractedData/'`

2. Modify Line 25 in the script to specify the path of the training data used in model training:

Line 25: `ref_data = pd.read_csv("./trainingdata.csv") **(Specify the path of training data used in the model training)**

3. Modify Line 48 in the script to specify the path for saving the network result:

Line 48: `name='./PredictedData/'+filename[:-4]+'.csv'` 
 
Script outputs:

Predict_.csv: *.csv* file containing the GCNet's output of each cell from cases WSI. Carcinoma cells (that exhibit cancerous characteristics) are classified as "positive", while non-carcinoma cells (that do not exhibit cancerous characteristics) are classified as "negative"

### Generation of Cancer Probability Heatmap

Step 1: Run the script `Contour_Line\HeatMap.py`.

1. Modify Line 65 in the script to specify the path of the exported WSI images:

Line 65: `img2 = Image.open(r'./original/'+file.replace('csv','png')).convert("RGBA")`

2. Modify Line 15 in the script to specify the path of GCNet's output of each cell from cases WSI after running the previous step:

Line 15: `path=r'./Contour_Line/PredictedData/'`

3. Modify Line 81 in the script to specify the path for saving the images with heatmap:

Line 81: `img2.save('./ExportHeatmap/'+file.replace('csv','png'))` 

Script outputs:

Heatmap.png: *.png* file containing the heatmap of the WSI image from yellow to black. **(Yellow color indicates relative high density of carcinoma cell, while black color indicates relative low density of carcinoma cell.)** 

### Generation of computer-aided diagnosis with contour for highlighting carcinoma regions

Step 1. Run the script `Contour_Line\ContourLineOverlay.py`.

1. Modify Line 64 in the script to specify the path of the exported WSI images:

Line 64: `img2 = Image.open(r'./Contour_Line/original/'+file.replace('csv','png')).convert('L').convert('RGB')` 

2. Modify Line 12 in the script to specify the path of the GCNet's output of each cell from cases WSI after running the previous step:

Line 12: `path=r'./PredictedData/'` 

3. Modify Line 96 in the script to specify the path for saving the images with contour for highlighting carcinoma regions:

Line 96: `img2.save('./ExportImage2/'+file.replace('csv','png'))` 

Script outputs:

Contour Line.png: *.png* file containing the combination of grayscale WSI image with contour for highlighting carcinoma regions.

By following these steps, you will be able to generate cancer probability heatmap and computer-aided diagnosis with contour. 

## Network Running time

The running time of the provided scripts with the provided data in the external dataset may vary based on different environments. The following running time information was measured on a system with the following compututational specifications:

1. System: Intel® Core™ i9-9900K CPU at 3.6GHz
2. RAM: 32GB
3. GPU: NVIDIA GeForce RTX 3080

The running time for each stage of the system using the provided data is presented in the table below:

| Stage of the System | Running Time (s) |
| ------------- | ------------- |
| Network Running Time (GCNet with WSI Data)  | 8.5837  |
| Network Running Time (GCNet with Tiled Image Data)  | 36.2036  |
| Network Running Time (GastroFLOW)  | 23.5408  |
| Post-Processing Time (GCNet with WSI Data)  | 0.4937  |
| Post-Processing Time (GCNet with Tiled Image Data)  | 21.6656  |
| Post-Processing Time (GastroFLOW)  | 9.4012  |
| Total Time (GCNet with WSI Data)  | 9.0774  |
| Total Time (GCNet with Tile Data)  | 57.8692  |
| Total Time (GastroFLOW)  | 32.942  |
