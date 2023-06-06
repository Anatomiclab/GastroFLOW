# Accelerating gastric carcinoma diagnosis with a weakly supervised artificial intelligence-based system by prioritizing gastric biopsy whole slide images

This repository provides scripts (Gastric Carcinoma classification Network "GCNet" and Gastric-specific Prioritization Workflow "GastroFLOW") for the article "Accelerating gastric carcinoma diagnosis with a weakly supervised artificial intelligence-based system by prioritizing gastric biopsy whole slide images".

GCNet is the Gastric Carcinoma classification Network developed in the study. It is an ensemble of optimized Multilayer Perceptron (MLP) neural networks trained on downsampled cellular features extracted from gastric biopsy whole slide images (WSIs). GCNet is designed to classify WSIs as either carcinoma (CA) or non-carcinoma (non-CA) cases. The top-performing MLP networks, based on Area Under the Curve (AUC) values, are selected and ensembled together. A majority voting scheme is used to classify WSIs based on the sum of votes from the MLP-based networks. The average of the MLP network outputs is used to calculate the mean case malignancy score, which is a measure of the probability of a case being CA. This score is used to prioritize cases for further review.

GastroFLOW, on the other hand, is the Gastric Case Prioritization Workflow developed in the study. It integrates GCNet and additional processing steps to enhance the triage ability for CA cases, especially those with low tumor area content. GastroFLOW involves the tiling of non-CA WSIs into smaller image tiles and re-evaluation using GCNet. Tiled images with a mean malignancy prediction score above a certain threshold are considered positive for carcinoma (P-CA). Cases are then assigned a grading based on the original GCNet prediction and the P-CA prediction from the tiled images. Cases graded as CA or suspicious for carcinoma are prioritized for further review, while those graded as benign are deemed lower risk. GastroFLOW aims to improve the detection and prioritization of CA cases, particularly those with low tumor content.


## Content

[Data for Cross-Validation, External Validation](#data-for-cross-validation-external-validation)

[Data Statistics for Cross-Validation, External Validation](#data-statistics-for-cross-validation-external-validation)

[Python Environment Setup](#python-environment-setup)

[Export WSI and extract cellular features from WSI](#export-wsi-and-extract-cellular-features-from-wsi)

* [Export the WSI](#export-the-wsi)

* [Extract the Cellular Features](#extract-the-cellular-features)

[Generation of WSI data](#generation-of-wsi-data)

[Generation of downsampled WSI and its tiled image data for training and validation](#generation-of-downsampled-wsi-and-its-tiled-image-data-for-training-and-validation)

[Machine Learning Algorithms](#machine-learning-algorithms)

* [Cross-Validation](#cross-validation)

* [External Dataset, Generate score for Internal Dataset](#external-dataset-generate-score-for-internal-dataset)

[Finding the optimized parameters for models using Talos](#finding-the-optimized-parameters-for-models-using-talos)

[MLP Network Training for GCNet](#mlp-network-training-for-gcnet)

[GCNet in Cross-Validation](#gcnet-in-cross-validation)

[External Validation](#external-validation-1)

* [GCNet with WSI Data](#gcnet-with-wsi-data)

* [GCNet with Tile Data](#gcnet-with-tile-data)

* [GastrolFlow](#gastrolflow)

[Post-Processing](#post-processing)

* [Cross-Validation for GCNet](#cross-validation-for-gcnet)

* [External Validation](#external-validation-1)

* [GCNet with WSI Data](#gcnet-with-wsi-data-1)

* [GCNet with Tile Data](#gcnet-with-tile-data-1)

* [GastrolFlow](#gastrolflow-1)

[Generation of Contour Line and Heatmap](#generation-of-contour-line-and-heatmap)

* [Data for generating contour line and heatmap](#data-for-generating-contour-line-and-heatmap)

* [Step before generating contour line and heatmap](#step-before-generating-contour-line-and-heatmap)

* [Generating contour line](#generating-contour-line)

* [Generating Heatmap](#generating-heatmap)

[Running time](#running-time)

## Data for Model training, cross-Validation, external Validation, and retrospectective study

* Training Data: [Training Data](https://connectpolyu-my.sharepoint.com/:x:/g/personal/21118855r_connect_polyu_hk/EQ16M5yOAvtAiuQGKHwtIagBaHGgOwICKB6DLU8fUc2usQ?e=NmtRhD)

* Cross-Validation Data: [Internal Data](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EYlJePFwtM1GpSknK0adq18BDO7zwOF63QHHfGkmQqa9Xw)

* External Validation Data: [External Data](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EV7M2OkvBV5OlJQCqXtC_vgBlvTacytjn1yn9ptzf5gqfg)

* Retrospctive: [Retrospective Set](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EYtw2btVoQNOgcdiw3gCu4oBOayIsqlA6Ek0gQzljDWotA)


## Data Statistics for Cross-Validation, External Validation

| Dataset | Benign  | Malignant  | Benign to Malignant ratio  | Cases  | WSIs  |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Internal Data  | 407(62.71%) | 242(37.29%) | 1.68:1 | 649 | 2064 |
| External Data  | 222(71.15%) | 90(28.85%) | 2.46:1 | 312 | 739 |
| Retrospective Case  | 60(66.67%) | 30(33.33%) | 2:1 | 90 | 113 |

## Python Environment Setup

To reproduce the Python environment used in the project, please use the environment file is named `environment_tf2.yml` and is based on the Anaconda environment.

By following this step, you will be able to set up the Python environment with the required dependencies and packages used in the project.

![Image](Figure_QuPath.png)

Step 1: Export the WSI
1. Open the Software QuPath(0.2.0-m8)[Link](https://qupath.github.io/) and access the script editor by navigating to Automate -> Show script editor.
2. Load the script for image extraction located at `QuPath_Script\ImagesExport.groovy`.
3. Modify Line 7 in the script to specify the path where the extracted images will be saved: 

Line 7: `path="images/"` //**(Specify the path to store the extracted images)** 

Script Outputs:

(WSI Images).png: A .png file containing the extracted WSI images

Step 2: Extract the Cellular Features
1. Open the script editor in QuPath software (version 0.2.0-m8).
2. Load the script for cellular feature extraction located at `QuPath_Script\CellularFeaturesExtractionforWholeSlide.groovy.`
3. Modify Line 92 in the script to specify the path where the extracted features will be saved:

Line 92: `save_path = "Feature/"`  //**(Specify the path for saving the extracted features)** 

Script Outputs:

(WSI cellular features).txt: A .txt file containing the extracted cellular features for the WSI images.

By following these steps, you will be able to export the desired WSI images and extract cellular features using the provided QuPath scripts.

## WSI Data Generation

Once you have extracted the cellular features, you can generate Whole Slide Image (WSI) data using the provided Python script, `DataAggregationforWholeSlides.py`. Follow the steps below:

Step 1: Open the Python script, `DataAggregationforWholeSlides.py`.

Step 2: Modify Line 7 in the script to specify the path of the cellular features extracted from QuPath:

Line 7: `path=r'./RawData'`//**(Specify the path to the extracted cellular features)** 

Step 3: Modify Line 8 in the script to specify the path where the WSI data will be saved:

Line 8: `outPath=r'./aggregateddata.csv'`//**(Specify the path for saving the WSI data)** 

Script Outputs:

aggregateddata.csv: This is a .csv file that contains the generated WSI data.This file will include relevant information derived from the averaged cellular features of the WSIs.

By following these steps, you will be able to generate the WSI data using the provided Python script. The resulting aggregateddata.csv file will contain the relevant information derived from the averaged cellular features of the WSIs.


## Generation of tiled image data

To generate tiled image data for training and validation, follow the steps below:

Step 1: Run `QuPath_Script\stage_2.py`
1. Open the script `stage_2.py` located in the QuPath_Script directory.
2. Modify Line 9 in the script to specify the path where the cellular features of the WSIs generated from QuPath are stored:

Line 9: `feat_path = "./2022Gastrointernaldataraw/RAW_TXT-SET_20200520/"` **(Specify the path for the extracted cellular features)** 

3. Modify Line 17 in the script to specify the immediate path for saving the tile data:

Line 17: `saving_path="./Training_tile/"` **(Specify the immediate path for saving the tile data)**

4. Adjust the tile ratio as needed. By default, the extracted tile size is set to 500x500 pixels.

Line 10: `tile_ratio = [500]` **(Specify the desired tile size)**

Step 2: Run `QuPath_Script\tile.py`
1. Open the script `tile.py` located in the QuPath_Script directory.
2. Modify Line 9 in the script to specify the path of the saving path used in Step 1:

Line 9: `tile_path = "./Training_tile/RAW_TXT-SET_20200520/"` **(Fill in the immediate path from Step 1)**

3. Modify Line 20 in the script to specify the path for saving the final tile data:

Line 20: `df.to_csv("./RAW_TXT-SET_20200520_tiledata_{}.csv".format(r), index=False)` **(Fill in the path for saving the final tile data)**

Script Outputs:

(WSI cellular features)_tiledata.csv: This is a .csv file containing the extracted 41 features for the tile image data.

By following these steps, you will be able to generate tiled image data for training and validation purposes. The resulting tiledata.csv file will contain the extracted cellular features for the tile image data.


## Cross-Validation and External Validation using Machine Learning Algorithms

This section provides the necessary scripts to perform cross-validation on the internal dataset and test machine learning algorithms on the external dataset.

### Cross-Validation

To perform cross-validation using machine learning algorithms, follow these steps:

Step 1: run `machinelearn_internal.py`

1. Modify the following lines in the script to specify the data paths. Note that the cross-validation fold needs to be modified manually:

Line 111: `Train_data_path="801010/"` **(Specify the root path of the internal data)**

Line 116: `data = pd.read_csv(Train_data_path+'train_cross10.csv')` **(Specify the file name of the training data)**

Line 133: `data2 = pd.read_csv(Train_data_path+'test_cross10.csv')` **(Specify the file name of the testing data)**

2. Modify Line 112 in the script to specify the path for saving the results:

Line 112: `Result_path=""`**(Specify the path for saving the results)**

Script output:

(Machine Learning Algorithm).csv: A .csv file containing the predicted results of the machine learning models for the internal data.

Step 2: Run `Machine_learning_internal_generation.py`

1. Modify the following lines to specify the data paths. Note that the cross-validation fold needs to be modified manually:

Line 4: `csv_data=pd.read_csv(r"Internal/K-SVM.csv")` **(Specify the path of the predicted results from Step 1)**

Line 17: `csv_data=pd.read_csv(r"801010/final_training_gt.csv")` **(Specify the path of the ground truth)**

2. Modify the following line to specify the path for saving the results:

Line 76: `final.to_csv(r"internal_result/K-SVM.csv")`

Script outputs:

(Machine Learning Algorithm).csv: A .csv file containing the triage list of machine learning models for the internal data.

Note: This section does not include malignancy prediction scores. If you want to generate malignancy prediction scores for AUC calculation or prioritize cases for triaging, please refer to the "External Dataset, Internal Dataset for Score" section.


### Generation of malignancy prediction scores for external dataset

This section provides instructions for running machine learning algorithms on the external dataset to generate scores. Follow the steps below:

Step 1: Run `machinelearn_external_score.py`

1. Modify the following lines in the script to specify the data paths. Note that the cross-validation fold needs to be modified manually:

Line 116: `data = pd.read_csv('trainingdata.csv')` **(Specify the path of the training data)**

Line 135: `data2 = pd.read_csv('data/External_SlideData_gt.csv')` **(Specify the path of the external dataset containing the WSIs' cellular features with modified (Fake) ground truth.)**

Line 277: `data = pd.read_csv('data/External_SlideData.csv')` **(Specify the path of the external dataset containing the WSIs' cellular features)**

Line 279: `ref_data = pd.read_csv("data/GroundTruth_External.csv")` **(Specify the path of the ground truth in the external validation dataset)**

2. Modify the following line to specify the path for saving the results:

Line 114: `result_path="external/"`

Script Outputs:

(Machine Learning Algorithm).csv: A .csv file containing the predicted results of machine learning models for the external data.

Step 2: Run `Machine_learning_external_generation.py`

1. Modify the following lines to specify the data paths. Note that the cross-validation fold needs to be modified manually:

Line 4: `csv_data=pd.read_csv(r"SVM--linear/report_pos_.csv")` **(Specify the path of the results for the WSIs predicted as positive (CA) by the model)**

Line 13: `csv_data=pd.read_csv(r"data/GroundTruth_External.csv")` **(Specify the path of the ground truth in the external validation dataset)**

Line 63: `csv_data=pd.read_csv(r"SVM--linear/requireTile_.csv")` **(Specify the path of the results for the WSIs predicted as negative (non-CA) by the model)**

2. Modify the following line to specify the path for saving the results:

Line 115: `final.to_csv(r"external_result/SVM--linear.csv")`

Script Outputs:

(Machine Learning Algorithm).csv: A .csv file containing the list of machine learning model predictions for the external data.

To generate the triage list, sort the predictions in Excel based on the predicted label (Positive, Suspicious of Positive, Negative) and malignancy prediction scores.



## Hyperparameter Optimization using Talos

To find the optimized parameters for multilayer perceptron (MLP) models using Talos, follow the steps below:

Step 1: Run `talos_tunning.py` script

1. Modify the following line to specify the path for the training data:

Line 72: `data = pd.read_csv("trainingdata.csv")` **(Specify the path of the training data)**

Script Output:

tuner_{int(time.time())}.pkl: A .pkl file that contains the optimized parameters, along with the corresponding loss value and validation metrics. The results are stored in a CSV format. The CSV file provides the optimized hyperparameter settings for the MLP networks to develop GCNet.



## MLP Network Training

To train any MLP network used for ensembling as GCNet, follow the steps below using the ModelTraining.py script. Ensure that you have set the optimized parameters obtained from Talos to train MLP network.

Step 1: Run the `ModelTraining.py` script

1. Modify the following line in the script to specify the path for the training data:

Line 197: `data = pd.read_csv("trainingdata.csv")` **(Modify the following line in the script to specify the path for the training data:)**

Script Outputs:

(TPR, TNR, PPV, NPV, Accuracy, roc_auc, val_accuracy).h5: An .h5 file that serves as the model and its performance after training. This file contains the trained MLP network.

SummaryOfPerformance.csv: A .csv file that provides the performance metrics of the model for different rounds of training.

By following these steps and running the ModelTraining.py script, you will be able to train the MLP networks, and choose suitable MLP networks ensembling as GCNet. The optimized MLP networks used for ensembling as GCNet are stored at `model_file`.


## GCNet Performance Validation in Cross-Validation

This section provides the necessary script, `Validation_internal.py`, to validate the performance of GCNet in cross-validation.

To validate the performance of GCNet in cross-validation, follow these steps:

Step 1: Run `Validation_internal.py`

1. Modify the following lines in the script to specify the path after downloading the internal data:

Line 26: `ref_data = pd.read_csv("801010/train_cross10.csv")` **(Specify the file name of the training data)**

Line 44: `data = pd.read_csv("801010/test_cross10.csv")` **(Specify the file name of the testing data)**

2. Modify the following lines to specify the path for saving the results:

Line 146,147: `posName = 'Internal_cross10_report_pos_' +  '.csv'`,`negName = 'Internal_cross10_requireTile_' + '.csv'`

Line 162,163: `report_pos2.to_csv("Network_result/" + posName)`,`report_neg2.to_csv("Network_result/" + negName)`

Script Outputs:

report_pos_.csv: A .csv file containing the WSIs predicted as positive (CA) by the GCNet model.

requireTile_.csv: A .csv file containing the WSIs predicted as negative (non-CA)by the GCNet model.

After generating the predicted results, please refer to the "Post Processing" section to generate the triage list based on the results.




## External Validation

In order to validate the performance of GCNet and GastroFlow in External Validation, it divides into three parts:



### GCNet with WSI Data

Use script: `Validation_stage1.py`

**Modified Line(please fill the path after downloading the External Data.)**:

Line 27: `ref_data = pd.read_csv("trainingdata.csv")` **(Fill the file name of the training data)**

Line 47: `data = pd.read_csv("data/External_SlideData.csv")` **(Fill the file name of the testing data)**

**Modified Line(please fill the path for saving the results.)**:

Line 150,151: `posName = 'External_report_pos_' +  '.csv'`,`negName = 'External_requireTile_' + '.csv'`

Line 166,167: `report_pos2.to_csv("Network_result/" + posName)`,`report_neg2.to_csv("Network_result/" + negName)`

**Script outputs**:

* **report_pos_.csv**: *.csv* file which contains the WSI prediction in positive.

* **requireTile_.csv**: *.csv* file which contains the WSI prediction in negative.

After generate the predicted result, to generate the triage list of result, please follow the section **Post Processing**.



### GCNet with Tile Data

Use script: `Validation_stage2.py`

**Modified Line(please fill the path after downloading the External Data.)**:

Line 27: `ref_data = pd.read_csv("trainingdata.csv")` **(Fill the file name of the training data)**

Line 44: `data = pd.read_csv("data/External_tiledata_500.csv")` **(Fill the file name of the testing data)**

**Modified Line(please fill the path for saving the results.)**:

Line 69: `table.to_csv("Network_result/"+ "External_tiledata_500.csv")`

**Script outputs**:

* **"External_tiledata_500.csv"**: *.csv* file which contains the predicted result of Tiled Data.

After generate the predicted result, to generate the triage list of result, please follow the section **Post Processing**.



### GastrolFlow

Use script: `Validation_stage1_2.py`

**Modified Line(please fill the path after downloading the External Data.)**:

Line 27: `ref_data = pd.read_csv("trainingdata.csv")` **(Fill the file name of the training data)**

Line 47: `data = pd.read_csv("data/External_SlideData.csv")` **(Fill the file name of the WSI testing data)**
 
Line 180: `data = pd.read_csv("data/External_tiledata_500.csv")` **(Fill the file name of the tiled testing data)**

**Modified Line(please fill the path for saving the results.)**:

Line 150,151: `posName = 'External_report_pos_' +  '.csv'`,`negName = 'External_requireTile_' + '.csv'`

Line 166,167: `report_pos2.to_csv("Network_result/" + posName)`,`report_neg2.to_csv("Network_result/" + negName)`

Line 230: `table.to_csv("Network_result/"+ "External_tiledata_500.csv")`

**Script outputs**:

* **report_pos_.csv**: *.csv* file which contains the WSIs model predict positive.

* **requireTile_.csv**: *.csv* file which contains the WSIs model predict negative.

* **External_tiledata_500.csv**: *.csv* file which contains the predicted result of Tiled Data for the WSIs model predict negative.

After generate the predicted result, to generate the triage list of result, please follow the section **Post Processing**.

 
## Post-Processing

After predicting the WSIs and tiled data, to generate the triage lists for analysis, please follow the steps below:



### Cross-Validation for GCNet

Use script: `Post_processing\Slide_internal_Stage1_generation.py`

**Modified Line(please fill the path after running the prediction.)**:

Line 4: `csv_data=pd.read_csv(r"Network_result/Internal_cross10_report_pos_.csv"))` **(Fill the path of the result for the WSIs model predicted positive)**

Line 14: `csv_data=pd.read_csv(r"801010/final_training_gt.csv")` **(Fill the path of the ground truth in internal validation dataset)**

Line 64: `csv_data=pd.read_csv(r"Network_result/Internal_cross10_requireTile_.csv")` **(Fill the path of the result for the WSIs model predicted negative)**


**Modified Line(please fill the path for saving the results.)**:

Line 118: `final.to_csv(r"stage1_result/Internal_cross10_test_PN.csv")`

**Script outputs**:

* **Internal_.csv**: *.csv* file which contains the triage list for cross-validation test data. The file contains predicted score, predicted label, ground-truth label and items for confusion matrix (True Positive, False Positive, True Negative, False Negative).




### External Validation 

#### GCNet with WSI Data

Use script: `Post_processing\Slide_Stage1_generation.py`

**Modified Line(please fill the path after running the prediction.)**:

Line 8: `csv_data=pd.read_csv(r"Network/External_report_pos_.csv")` **(Fill the path of the result for the WSIs model predicted positive)**

Line 18: `csv_data=pd.read_csv(r"data/GroundTruth_External.csv")` **(Fill the path of the ground truth in external validation dataset)**

Line 66: `csv_data=pd.read_csv(r"Network/External_requireTile_.csv")` **(Fill the path of the result for the WSIs model predicted negative)**

**Modified Line(please fill the path for saving the results.)**:

Line 120: `final.to_csv(r"stage1_result/External_stage1_test_PN.csv")`

**Script outputs**:

* **External__.csv**: *.csv* file which contains the triage list for External test data based on WSI data. The file contains predicted score, predicted label, ground-truth label and items for confusion matrix (True Positive, False Positive, True Negative, False Negative).

To generate **the prioritization of cases**, please use Excel function to do.





#### GCNet with Tile Data

Please follow the steps below:

**Step 1**: Run script: `Post_processing\Tile_Calc_Avg_score.py`

**Modified Line(please fill the path after running the prediction.)**:

Line 10: `folder_path=r"Network_result//"` **(Fill the path of the predicted result after running the network prediction)**

Line 12: `csv_data=pd.read_csv(folder_path+r"External_tiledata_500.csv")` **(Fill the filename of the predicted result after running the network prediction)**

**Modified Line(please fill the path for saving the results.)**:

Line 42: `final.to_csv(r"avg_score//"+"External_tiledata_500.csv")` **("avg_score//" is immediate path. Please Create it manually when the path is not existed.)**

**Script outputs**:

* **avgscore.csv**: *.csv* file which contains the average predicted score of 11 models for each tile data.

**Step 2**: Run script: `Post_processing\Tile_Case_split_ver2.py`

**Modified Line(please fill the path after running the Step1.)**:

Line 11,13: `folder_path=r"avg_score/"`,`csv_data=pd.read_csv(folder_path+r"External_tiledata_500.csv")` **(Fill the immediate path in step 1)** 

**Modified Line(please fill the path for saving the results.)**:

Line 82,83,87: `#shutil.rmtree(r"case_split//")`, `os.makedirs(r"case_split//",exist_ok=True)`, `case_final.to_csv(r"case_split//"+str(i)+".csv",index=False)` **("case_split//" is immediate path. Please Create it manually when the path is not existed.)**

**Script outputs**:

* **Folder of Case.csv**: *.csv* file which contains the predicted result of tiled data corresponding to each cases.

**Step 3**: Run script: `Post_processing\Tile_Case_Sort.py`

**Modified Line(please fill the path after running the Step2.)**:

Line 15: `for filename in glob.glob(r"case_split/*"):` **(Fill the immediate path in step 2)**

**Modified Line(please fill the path for saving the results.)**:

Line 12,13: `#shutil.rmtree("case_split_Sorted//")"`,`os.makedirs("case_split_Sorted//",exist_ok=True)` **("case_split_Sorted//" is immediate path. Please Create it manually when the path is not existed.)**

**Script outputs**:

* **Folder of Case.csv**: *.csv* file which contains the predicted result of tiled data corresponding to each cases, after this step, the score will be sorted in decreasing order.

**Step 4**: Run script: `Post_processing\Tile_Case_Combine_ver1.py`

**The threshold for prediction**

Line 9: `threshold=0.2`

**Modified Line(please fill the path after running the Step3.)**:

Line 18: `for filename in glob.glob(r"case_split_Sorted/*"):` **(Fill the immediate path in step 3)**

**Modified Line(please fill the path for saving the results.)**:

Line 53,57: `os.makedirs("Final_Tile_Stage2//", exist_ok=True)`,`case_final.to_csv(r"Final_Tile_Stage2//External_" +"_"+str(threshold)+ ".csv", index=False)` **("Final_Tile_Stage2//" is immediate path)**

**Script outputs**:

* **External_.csv**: *.csv* file which contains the predicted result for External test data. It contains the average score, the minimum number of positive tile to become the suspicious positive, the actual number of positive tile and the diagnosis.

**Step 5**: Run script: `Post_processing\Tile_Case_Stage2_generation.py`

**Modified Line(please fill the path after running the Step4.)**:

Line 9: `csv_data=pd.read_csv(r"Final_Tile_Stage2/External_500_"+str(threshold)+".csv")` **(Fill the immediate path in step 4)**

Line 19: `csv_data=pd.read_csv(r"data/GroundTruth_External.csv")`  **(Fill the path of ground truth in external dataset)**

**Modified Line(please fill the path for saving the results.)**:

Line 78: `final.to_csv(r"stage2_result/External_stage2_test_PN_"+str(threshold)+".csv")`

**Script outputs**:

* **External_.csv**: *.csv* file which contains the triage list for External test data based on tiled data.The file contains predicted score, predicted label, ground-truth label and items for confusion matrix (True Positive, False Positive, True Negative, False Negative).

To generate **the prioritization of cases**, please use Excel function to do.
(To generate **the prioritization of cases**, please use Excel to sort by classification label(carcinoma, suspicious of carcinoma, benign) then sort by prediction score)




#### GastrolFlow

Before the step, please run the step **GCNet with Tile Data** using the generated result of tiled data from `Validation_stage1_2.py`:

After the step, it should generate the triage list for tiled data while the corresponding WSI data is predicted as negative.

Then, please follow the steps:

**Step 1**: Run script: `Post_processing\Slide_Tile_Filter_Temporary_Solution.py`

**Modified Line(please fill the path after running the Step3.)**:

Line 7: `stage1_csv=pd.read_csv(r"Network/External_requireTile_.csv")` **(Fill the path of the result for the WSIs model predicted negative)**

Line 10: `stage1_csv_pos=pd.read_csv(r"Network/External_report_pos_.csv")`**(Fill the path of the result for the WSIs model predicted positive)**

Line 20: `filename=r"Final_Tile_Stage2/External_500_"+str(threshold)+".csv"` **(Fill the path of the result for the tile data)**

**Modified Line(please fill the path for saving the results.)**:

Line 52: `case_final.to_csv(r"Final_Tile_Stage1_2/"+filename.split('/')[-1],index=False)` **("Final_Tile_Stage1_2//" is immediate path)**

**Script outputs**:

* **External_.csv**: *.csv* file which contains the predicted result for external dataset. It will provide the prediction label (positive, suspect positive and negative), the predict score, and the number of the tiled predicted as suspect positive.

**Step 2**: Run script: `Post_processing\Slide_Tile_Stage1_2_generation.py`

**Modified Line(please fill the path after running the Step4.)**:

Line 6: `threshold=0.2` **(Fill the immediate path in step 1)**

Line 8: `csv_data=pd.read_csv(r"Final_Tile_Stage1_2/External_500_"+str(threshold)+".csv")` **(Fill the immediate path in step 1)**

Line 18: `csv_data=pd.read_csv(r"data/GroundTruth_External.csv")` **(Fill the path of ground truth in external dataset)**

**Modified Line(please fill the path for saving the results.)**:

Line 82: `final.to_csv(r"stage1_2_result/External_stage1_2_test_PN_"+str(threshold)+".csv")`

**Script outputs**:

* **External_.csv**: *.csv* file which contains the triage list for External test data based on WSI and tiled data.The file contains predicted score, predicted label, ground-truth label and items for confusion matrix (True Positive, False Positive, True Negative, False Negative).

To generate the triage list of cases, please use excel to sort according the prediction label (Positive, Suspect Positive and Negative), and the prediction score.




## Generation of Contour Line and Heatmap



### Data for generating contour line and heatmap

We provide the sample data for validating the scripts.

The data could be downloaded at [Link](Sample_Feature_For_Generating_Contour_Line.txt).

Please note that, we provide both WSI images and corresponding cellular features. Therefore, you could directly run the following steps.

Sample of WSI images is shown below:

![Image](Sample_WSI.png)



### Step before generating contour line and heatmap

Before generating the maps, please run the script `Contour_Line\AppendingPreductionToCells.py` first.

**Modified Line(please fill the path for your WSI image and extracted cellular features.)**:

Line 20: `path=r'./Contour_Line/ExtractedData/'` **(Fill the path of the extracted cellular features)**

**Modified Line for Code Running**: 

Line 25: `ref_data = pd.read_csv("./trainingdata.csv") #Fill the Training Data path`**(Fill the path of training data used in the model training)**

**Modified Line(please fill the path for saving the results.)**:

Line 48: `name='./PredictedData/'+filename[:-4]+'.csv'` **(Fill the path for saving the network result)**
 
**Script outputs**:

* **Predict_.csv**: *.csv* file which contains the network output result based on the provided cellular features.




### Generating contour line

Run the script `Contour_Line\ContourLineOverlay.py`.

**Modified Line(please fill the path for your WSI image.)**:

Line 64: `img2 = Image.open(r'./Contour_Line/original/'+file.replace('csv','png')).convert('L').convert('RGB')` **(Fill the path of the extracted WSI images)**

**Modified Line(please fill the path after running the step before generation.)**:

Line 12: `path=r'./PredictedData/'` **(Fill the path of the network result in step before generating contour line and heatmap)**

**Modified Line(please fill the path for saving the results.)**:

Line 96: `img2.save('./ExportImage2/'+file.replace('csv','png'))` **(Fill the path for saving the images with contour lines)**

**Script outputs**:

* **Contour Line.png**: *.png* file which contains the combination of grayscale WSI image and contour lines.



### Generating Heatmap

Run the script `Contour_Line\HeatMap.py`.

**Modified Line(please fill the path for your WSI image.)**:

Line 65: `img2 = Image.open(r'./original/'+file.replace('csv','png')).convert("RGBA")` **(Fill the path of the extracted WSI images)**

**Modified Line(please fill the path after running the step before generation.)**:

Line 15: `path=r'./Contour_Line/PredictedData/'` **(Fill the path of the network result in step before generating contour line and heatmap)**

**Modified Line(please fill the path for saving the results.)**:

Line 81: `img2.save('./ExportHeatmap/'+file.replace('csv','png'))` **(Fill the path for saving the images with heatmap)**

**Script outputs**:

* **Heatmap.png**: *.png* file which contains the combination of grayscale WSI image and heatmap.




## Running time

We calculate the running time of using the scripts with our provided data in external dataset. Please note that, based on different environment, the time will be slightly different.

| The Stage of the System  | Running Time(s)  |
| ------------- | ------------- |
| Network Running Time (GCNet with WSI Data)  | 8.5837  |
| Network Running Time (GCNet with Tile Data)  | 36.2036  |
| Network Running Time (GastrolFlow)  | 23.5408  |
| Post-Processing Time (GCNet with WSI Data)  | 0.4937  |
| Post-Processing Time (GCNet with Tile Data)  | 21.6656  |
| Post-Processing Time (GastrolFlow)  | 9.4012  |
| Total Time (GCNet with WSI Data)  | 9.0774  |
| Total Time (GCNet with Tile Data)  | 57.8692  |
| Total Time (GastrolFlow)  | 32.942  |
