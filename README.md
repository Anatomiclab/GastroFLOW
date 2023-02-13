# Prioritization on whole slide images of clinical gastric carcinoma biopsies through a weakly supervised and annotation-free system

This repository provides training and testing scripts for the article "Prioritization on whole slide images of clinical gastric carcinoma biopsies through a weakly supervised and annotation-free system".


## Content

[Environment of Python](#environment-of-python)

[Using QuPath to generate the WSI images and cellular features](#Using-QuPath-to-generate-the-WSI-images-and-cellular-features)

* [Extract the WSI Images](#extract-the-wsi-images)

* [Extract the cellular features](#extract-the-cellular-features)

[Generation of slide data](#generation-of-slide-data)

[Generation of tile data for training and validation](#generation-of-tile-data-for-training-and-validation)

[Data for Cross-Validation, External Validation](#data-for-cross-validation-external-validation)

[Data Statistics for Cross-Validation, External Validation](#data-statistics-for-cross-validation-external-validation)

[Machine Learning Algorithms](#machine-learning-algorithms)

* [Cross-Validation](#cross-validation)

* [External Dataset, Internal Dataset for Score](#external-dataset-internal-dataset-for-score)

[Finding the optimized parameters for models using Talos](#finding-the-optimized-parameters-for-models-using-talos)

[Network Training](#network-training)

[Cross-Validation](#cross-validation-1)

[External Validation](#external-validation-1)

* [GCNet with Slide Data](#gcnet-with-slide-data)

* [GCNet with Tile Data](#gcnet-with-tile-data)

* [GastrolFlow](#gastrolflow)

[Post-Processing for generating Triage List](#post-processing-for-generating-triage-list)

* [Cross-Validation](#cross-validation-2)

* [External Validation](#external-validation-2)

* [GCNet with Slide Data](#gcnet-with-slide-data-1)

* [GCNet with Tile Data](#gcnet-with-tile-data-1)

* [GastrolFlow](#gastrolflow-1)

[Generation of Contour Line and Heatmap](#generation-of-contour-line-and-heatmap)

* [Data for generating contour line and heatmap](#data-for-generating-contour-line-and-heatmap)

* [Step before generating contour line and heatmap](#step-before-generating-contour-line-and-heatmap)

* [Generating contour line](#generating-contour-line)

* [Generating Heatmap](#generating-heatmap)

[Running time](#running-time)


## Environment of Python

We update the environment file of Python using in the project. The anaconda environment file is `environment_tf2.yml`

Please follow the [import guideline](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to reproduce the environment.

## Using QuPath to generate the WSI images and cellular features

We use the QuPath(0.2.0-m8) [Link](https://qupath.github.io/) to generate the WSI images and cellular features for training and validation.

The capture of working space is shown below:

![Image](Figure_QuPath.png)

To run the scripts, please open the script editor via `Automate->Show script editor` and load the script for running.

If you want to extract the WSI images and cellular features, please follow the steps below:

### Extract the WSI Images

For extracting the images, use QuPath script `QuPath_Script\ImagesExport.groovy`.

**Modified Line(please fill the path for saving the extracted images)**:

Line 7: `path="images/" // Please Change the Path`  **(Path for saving the extracted images)**


**Script outputs**:

* **(WSI Images).png**: *.png* file which are the extracted WSI images.

### Extract the cellular features

For extracting the cellular feature, use QuPath script`QuPath_Script\CellularFeaturesExtractionforWholeSlide.groovy`.

**Modified Line(please fill the path for saving the extracted features)**:

Line 92: `save_path = "Feature/"  //CHANGE sve path here` **(Path for saving the extracted features)**

**Script outputs**:

* **(WSI cellular features).txt**: *.txt* file which contains the extracted cellular features for WSI images.



## Generation of slide data

After extracting the cellular features, please use the Python script `DataAggregationforWholeSlides.py` to generate the slide data.

**Modified Line(please fill the path for the path of cellular features extracted from QuPath)**:

Line 7: `path=r'./RawData'`

**Modified Line(please fill the path for saving the slide data)**:

Line 8: `outPath=r'./aggregateddata.csv'`

**Script outputs**:

* **aggregateddata.csv**: *.csv* file which is the slide data. 


## Generation of downsampled WSI and its tiled image data for training and validation

After using QuPath to generate the cellular features, if you want to generate the tile data for the GastrolFlow training and validation, please follow the steps below:

**Step 1**: run `QuPath_Script\stage_2.py`

**Modified Line(please fill the path for the path of cellular features of slides)**:

Line 9: `feat_path = "./2022Gastrointernaldataraw/RAW_TXT-SET_20200520/"` **(Path for the cellular features of slides from QuPath)**

**Modified Line(please fill the path for the path of saving)**:

Line 17: `saving_path="./Training_tile/"` **(Immediate path for saving the tile data)**

**Setting of the range of tile ratio**:

Line 10: `tile_ratio = [500]` **(The extracted tile is 500*500)**

**Step 2**: run `QuPath_Script\tile.py`

**Modified Line(please fill the path for the path of step 1 saving path)**:

Line 9: `tile_path = "./Training_tile/RAW_TXT-SET_20200520/"` **(Fill the immediate path in Step 1)**

**Modified Line(please fill the path for the path of saving)**:

Line 20: `df.to_csv("./RAW_TXT-SET_20200520_tiledata_{}.csv".format(r), index=False)` **(Fill the path for saving the final tile data)**

**Script outputs**:

* **(WSI cellular features)_tiledata.csv**: *.csv* file which contains the extracted 41 features for tile data.



## Data for Cross-Validation, External Validation

* Training Data: [Training Data](https://connectpolyu-my.sharepoint.com/:x:/g/personal/21118855r_connect_polyu_hk/EQ16M5yOAvtAiuQGKHwtIagBaHGgOwICKB6DLU8fUc2usQ?e=NmtRhD)

* Cross-Validation Data: [Internal Data](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EYlJePFwtM1GpSknK0adq18BDO7zwOF63QHHfGkmQqa9Xw)

* External Validation Data: [External Data](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EV7M2OkvBV5OlJQCqXtC_vgBlvTacytjn1yn9ptzf5gqfg)

* Retrospctive: [Retrospective Set](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118855r_connect_polyu_hk/EYtw2btVoQNOgcdiw3gCu4oBOayIsqlA6Ek0gQzljDWotA)


## Data Statistics for Cross-Validation, External Validation

| Dataset | Cases  | WSIs  |
| ------------- | ------------- | ------------- |
| Internal Data  | 649 | 2064 |
| External Data  | 312 | 739 |
| Retrospective Case  | 90 | 113 |


## Machine Learning Algorithms

We provide the running scripts of machine learning algorithms in Internal Set for Cross-Validation and tested in External dataset.



### Cross-Validation (Check)

To run the machine learning algorithms in Cross-Validation, please follow the steps below: 

**Step 1**: run `machinelearn_internal.py`

**Modified Line(please fill the path for the data. Please note that the Cross-Validation needs to modified the Fold manually)**:

Line 111: `Train_data_path="801010/"` **(Fill the root path of the internal data)**

Line 116, `data = pd.read_csv(Train_data_path+'train_cross10.csv')` **(Fill the file name of the training data)**

Line 133: `data2 = pd.read_csv(Train_data_path+'test_cross10.csv')` **(Fill the file name of the testing data)**

**Modified Line(please fill the path for saving the results.)**:

Line 112: `Result_path=""`

**Script outputs**:

* **(Machine Learning Algorithm).csv**: *.csv* file which contains the predicted results of machine learning models for internal data.

**Step 2**: run `Machine_learning_internal_generation.py`

**Modified Line(please fill the path for the data. Please note that the Cross-Validation needs to modified the Fold manually)**:

Line 4: `csv_data=pd.read_csv(r"Internal/K-SVM.csv")` **(Fill the path of the predicted result in the Step 1)**

Line 17: `csv_data=pd.read_csv(r"801010/final_training_gt.csv")` **(Fill the path of the ground truth)**

**Modified Line(please fill the path for saving the results.)**:

Line 76: `final.to_csv(r"internal_result/K-SVM.csv")`

**Script outputs**:

* **(Machine Learning Algorithm).csv**: *.csv* file which contains the triage list of machine learning models for internal data. 

In this section, it does not contain the scores. If you want to generate the score for calculating AUC or finding priorities, please turn to section **External Dataset, Internal Dataset for Score**.




### External Dataset, Generate score for Internal Dataset(Check)

To run the machine learning algorithms in External Dataset, or generating the score in Internal Dataset, please follow the steps below: 

**Step 1**: run `machinelearn_external_score.py`

**Modified Line(please fill the path for the data. Please note that the Cross-Validation needs to modified the Fold manually)**:

Line 116: `data = pd.read_csv('trainingdata.csv')` **(Fill the path of the training data)**

Line 135: `data2 = pd.read_csv('data/External_SlideData_gt.csv')` **(Fill the path of the features of slides in external validation dataset modified with (Fake) ground truth.)**

Line 277: `data = pd.read_csv('data/External_SlideData.csv')` **(Fill the path of the features of slides in external validation dataset)**

Line 279: `ref_data = pd.read_csv("data/GroundTruth_External.csv")` **(Fill the path of the ground truth in external validation dataset)**

**Modified Line(please fill the path for saving the results.)**:

Line 114: `result_path="external/"`

**Script outputs**:

* **(Machine Learning Algorithm).csv**: *.csv* file which contains the predicted results of machine learning models for external data.

**Step 2**: run `Machine_learning_external_generation.py`

**Modified Line(please fill the path for the data. Please note that the Cross-Validation needs to modified the Fold manually)**:

Line 4: `csv_data=pd.read_csv(r"SVM--linear/report_pos_.csv")` **(Fill the path of the result for the slides model predicted positive)**

Line 13: `csv_data=pd.read_csv(r"data/GroundTruth_External.csv")` **(Fill the path of the ground truth in external validation dataset)**

Line 63: `csv_data=pd.read_csv(r"SVM--linear/requireTile_.csv")` **(Fill the path of the result for the slides model predicted negative)**

**Modified Line(please fill the path for saving the results.)**:

Line 115: `final.to_csv(r"external_result/SVM--linear.csv")`

**Script outputs**:

* **(Machine Learning Algorithm).csv**: *.csv* file which contains the list of machine learning model predictions for external data. 

To generate the triage list, please use excel to sort according by predict label (Positive, Suspicous of Positive, Negative) and score.



## Finding the optimized parameters for models using Talos (Check)

To find the optimized parameters of model using Talos, use script `talos_tunning.py`

**Modified Line(please fill the path for the training data.)**:

Line 72: `data = pd.read_csv("trainingdata.csv")` **(Fill the path of the training data)**

**Script outputs**:

* **tuner_{int(time.time())}.pkl**: *.pkl* file which contains the finding parameters and the corresponding loss value and validation metrics.




## Network Training (Check)

To train a model, use script `ModelTraining.py`.

**Modified Line(please fill the path for the training data.)**:

Line 197: `data = pd.read_csv("trainingdata.csv")` **(Fill the path of the training data)**

**Script outputs**:

* **(TPR, TNR, PPV, NPV, Accuracy, roc_auc, val_accuracy).h5**: *.h5* file which is the model checkpoint after training.

* **SummaryOfPerformance.csv**: *.csv* file which is the performance of model in different rounds.




## GCNet in Cross-Validation (Check)

In order to validate the performance of GCNet in Cross-Validation, use script `Validation_internal.py`

**Modified Line(please fill the path after downloading the Internal Data.)**:

Line 26: `ref_data = pd.read_csv("801010/train_cross10.csv")` **(Fill the file name of the training data)**

Line 44: `data = pd.read_csv("801010/test_cross10.csv")` **(Fill the file name of the testing data)**

**Modified Line(please fill the path for saving the results.)**:

Line 146,147: `posName = 'Internal_cross10_report_pos_' +  '.csv'`,`negName = 'Internal_cross10_requireTile_' + '.csv'`

Line 162,163: `report_pos2.to_csv("Network_result/" + posName)`,`report_neg2.to_csv("Network_result/" + negName)`

**Script outputs**:

* **report_pos_.csv**: *.csv* file which contains the Slides model predict positive.

* **requireTile_.csv**: *.csv* file which contains the Slides model predict negative.

After generate the predicted result, to generate the triage list of result, please follow the section **Post Processing**.





## External Validation

In order to validate the performance of GCNet and GastroFlow in External Validation, it divides into three parts:



### GCNet with Slide Data (Check)

Use script: `Validation_stage1.py`

**Modified Line(please fill the path after downloading the External Data.)**:

Line 27: `ref_data = pd.read_csv("trainingdata.csv")` **(Fill the file name of the training data)**

Line 47: `data = pd.read_csv("data/External_SlideData.csv")` **(Fill the file name of the testing data)**

**Modified Line(please fill the path for saving the results.)**:

Line 166,167: `report_pos2.to_csv("Network_result/" + posName)`,`report_neg2.to_csv("Network_result/" + negName)`

**Script outputs**:

* **report_pos_.csv**: *.csv* file which contains the Slides model predict positive.

* **requireTile_.csv**: *.csv* file which contains the Slides model predict negative.

After generate the predicted result, to generate the triage list of result, please follow the section **Post Processing**.



### GCNet with Tile Data (Check)

Use script: `Validation_stage2.py`

**Modified Line(please fill the path after downloading the External Data.)**:

Line 27: `ref_data = pd.read_csv("trainingdata.csv")` **(Fill the file name of the training data)**

Line 44: `data = pd.read_csv("data/External_tiledata_500.csv")` **(Fill the file name of the testing data)**

**Modified Line(please fill the path for saving the results.)**:

Line 69: `table.to_csv("Network_result/"+ "External_tiledata_500.csv")`

**Script outputs**:

* **"External_tiledata_500.csv"**: *.csv* file which contains the predicted result of Tiled Data.

After generate the predicted result, to generate the triage list of result, please follow the section **Post Processing**.



### GastrolFlow (Check)

Use script: `Validation_stage1_2.py`

**Modified Line(please fill the path after downloading the External Data.)**:

Line 27: `ref_data = pd.read_csv("trainingdata.csv")` **(Fill the file name of the training data)**

Line 47: `data = pd.read_csv("data/External_SlideData.csv")` **(Fill the file name of the slide testing data)**
 
Line 180: `data = pd.read_csv("data/External_tiledata_500.csv")` **(Fill the file name of the tiled testing data)**

**Modified Line(please fill the path for saving the results.)**:

Line 166,167: `report_pos2.to_csv("Network_result/" + posName)`,`report_neg2.to_csv("Network_result/" + negName)`

Line 230: `table.to_csv("Network_result/"+ "External_tiledata_500.csv")`

**Script outputs**:

* **report_pos_.csv**: *.csv* file which contains the Slides model predict positive.

* **requireTile_.csv**: *.csv* file which contains the Slides model predict negative.

* **External_tiledata_500.csv**: *.csv* file which contains the predicted result of Tiled Data for the slides model predict negative.


 
## Post-Processing for generating Triage List

After predicting the slides and tiled data, to generate the triage lists for analysis, please follow the steps below:



### Cross-Validation (Check)

Use script: `Post_processing\Slide_internal_Stage1_generation.py`

**Modified Line(please fill the path after running the prediction.)**:

Line 4: `csv_data=pd.read_csv(r"Network_result/Internal_cross10_report_pos_.csv"))` **(Fill the path of the result for the slides model predicted positive)**

Line 14: `csv_data=pd.read_csv(r"801010/final_training_gt.csv")` **(Fill the path of the ground truth in internal validation dataset)**

Line 64: `csv_data=pd.read_csv(r"Network_result/Internal_cross10_requireTile_.csv")` **(Fill the path of the result for the slides model predicted negative)**


**Modified Line(please fill the path for saving the results.)**:

Line 118: `final.to_csv(r"stage1_result/Internal_cross10_test_PN.csv")`

**Script outputs**:

* **Internal_.csv**: *.csv* file which contains the triage list for cross-validation test data. The file contains predicted score, predicted label, ground-truth label and items for confusion matrix (True Positive, False Positive, True Negative, False Negative).




### External Validation 

#### GCNet with Slide Data (Check)

Use script: `Post_processing\Slide_Stage1_generation.py`

**Modified Line(please fill the path after running the prediction.)**:

Line 8: `csv_data=pd.read_csv(r"Network/External_report_pos_.csv")` **(Fill the path of the result for the slides model predicted positive)**

Line 18: `csv_data=pd.read_csv(r"data/GroundTruth_External.csv")` **(Fill the path of the ground truth in external validation dataset)**

Line 66: `csv_data=pd.read_csv(r"Network/External_requireTile_.csv")` **(Fill the path of the result for the slides model predicted negative)**

**Modified Line(please fill the path for saving the results.)**:

Line 120: `final.to_csv(r"stage1_result/External_stage1_test_PN.csv")`

**Script outputs**:

* **External__.csv**: *.csv* file which contains the triage list for External test data based on slide data. The file contains predicted score, predicted label, ground-truth label and items for confusion matrix (True Positive, False Positive, True Negative, False Negative).

To generate **the prioritization of cases**, please use Excel function to do.





#### GCNet with Tile Data

Please follow the steps below:

**Step 1**: Run script: `Post_processing\Tile_Calc_Avg_score.py` (Check)

**Modified Line(please fill the path after running the prediction.)**:

Line 10: `folder_path=r"Network_result//"` **(Fill the path of the predicted result after running the network prediction)**

Line 12: `csv_data=pd.read_csv(folder_path+r"External_tiledata_500.csv")` **(Fill the filename of the predicted result after running the network prediction)**

**Modified Line(please fill the path for saving the results.)**:

Line 42: `final.to_csv(r"avg_score//"+"External_tiledata_500.csv")` **("avg_score//" is immediate path. Please Create it manually when the path is not existed.)**

**Script outputs**:

* **avgscore.csv**: *.csv* file which contains the average predicted score of 11 models for each tile data.

**Step 2**: Run script: `Post_processing\Tile_Case_split_ver2.py` (Check)

**Modified Line(please fill the path after running the Step1.)**:

Line 11,13: `folder_path=r"avg_score/"`,`csv_data=pd.read_csv(folder_path+r"External_tiledata_500.csv")` **(Fill the immediate path in step 1)** 

**Modified Line(please fill the path for saving the results.)**:

Line 82,83,87: `#shutil.rmtree(r"case_split//")`, `os.makedirs(r"case_split//",exist_ok=True)`, `case_final.to_csv(r"case_split//"+str(i)+".csv",index=False)` **("case_split//" is immediate path. Please Create it manually when the path is not existed.)**

**Script outputs**:

* **Folder of Case.csv**: *.csv* file which contains the predicted result of tiled data corresponding to each cases.

**Step 3**: Run script: `Post_processing\Tile_Case_Sort.py` (Check)

**Modified Line(please fill the path after running the Step2.)**:

Line 15: `for filename in glob.glob(r"case_split/*"):` **(Fill the immediate path in step 2)**

**Modified Line(please fill the path for saving the results.)**:

Line 12,13: `#shutil.rmtree("case_split_Sorted//")"`,`os.makedirs("case_split_Sorted//",exist_ok=True)` **("case_split_Sorted//" is immediate path. Please Create it manually when the path is not existed.)**

**Script outputs**:

* **Folder of Case.csv**: *.csv* file which contains the predicted result of tiled data corresponding to each cases, after this step, the score will be sorted in decreasing order.

**Step 4**: Run script: `Post_processing\Tile_Case_Combine_ver1.py` (Check)

**The threshold for prediction**

Line 9: `threshold=0.2`

**Modified Line(please fill the path after running the Step3.)**:

Line 18: `for filename in glob.glob(r"case_split_Sorted/*"):` **(Fill the immediate path in step 3)**

**Modified Line(please fill the path for saving the results.)**:

Line 53,57: `os.makedirs("Final_Tile_Stage2//", exist_ok=True)`,`case_final.to_csv(r"Final_Tile_Stage2//External_" +"_"+str(threshold)+ ".csv", index=False)` **("Final_Tile_Stage2//" is immediate path)**

**Script outputs**:

* **External_.csv**: *.csv* file which contains the predicted result for External test data. It contains the average score, the minimum number of positive tile to become the suspicious positive, the actual number of positive tile and the diagnosis.

**Step 5**: Run script: `Post_processing\Tile_Case_Stage2_generation.py` (Check)

**Modified Line(please fill the path after running the Step4.)**:

Line 9: `csv_data=pd.read_csv(r"Final_Tile_Stage2/External_500_"+str(threshold)+".csv")` **(Fill the immediate path in step 4)**

Line 19: `csv_data=pd.read_csv(r"data/GroundTruth_External.csv")`  **(Fill the path of ground truth in external dataset)**

**Modified Line(please fill the path for saving the results.)**:

Line 78: `final.to_csv(r"stage2_result/External_stage2_test_PN_"+str(threshold)+".csv")`

**Script outputs**:

* **External_.csv**: *.csv* file which contains the triage list for External test data based on tiled data.The file contains predicted score, predicted label, ground-truth label and items for confusion matrix (True Positive, False Positive, True Negative, False Negative).

To generate **the prioritization of cases**, please use Excel function to do.




#### GastrolFlow

Before the step, please run the step **Only Tile Data** using the generated result of tiled data from `Validation_stage1_2.py`:

After the step, it should generate the triage list for tiled data while the corresponding slide data is predicted as negative.

Then, please follow the steps:

**Step 1**: Run script: `Post_processing\Slide_Tile_Filter_Temporary_Solution.py` (Check)

**Modified Line(please fill the path after running the Step3.)**:

Line 7: `stage1_csv=pd.read_csv(r"Network/External_requireTile_.csv")` **(Fill the path of the result for the slides model predicted negative)**

Line 10: `stage1_csv_pos=pd.read_csv(r"Network/External_report_pos_.csv")`**(Fill the path of the result for the slides model predicted positive)**

Line 20: `filename=r"Final_Tile_Stage2/External_500_"+str(threshold)+".csv"` **(Fill the path of the result for the tile data)**

**Modified Line(please fill the path for saving the results.)**:

Line 52: `case_final.to_csv(r"Final_Tile_Stage1_2/"+filename.split('/')[-1],index=False)` **("Final_Tile_Stage1_2//" is immediate path)**

**Script outputs**:

* **External_.csv**: *.csv* file which contains the predicted result for external dataset. It will provide the prediction label (positive, suspect positive and negative), the predict score, and the number of the tiled predicted as suspect positive.

**Step 2**: Run script: `Post_processing\Slide_Tile_Stage1_2_generation.py` (Check)

**Modified Line(please fill the path after running the Step4.)**:

Line 8: `csv_data=pd.read_csv(r"Final_Tile_Stage1_2/External_500_"+str(threshold)+".csv")` **(Fill the immediate path in step 1)**

Line 18: `csv_data=pd.read_csv(r"data/GroundTruth_External.csv")` **(Fill the path of ground truth in external dataset)**

**Modified Line(please fill the path for saving the results.)**:

Line 82: `final.to_csv(r"stage1_2_result/External_stage1_2_test_PN_"+str(threshold)+".csv")`

**Script outputs**:

* **External_.csv**: *.csv* file which contains the triage list for External test data based on slide and tiled data.The file contains predicted score, predicted label, ground-truth label and items for confusion matrix (True Positive, False Positive, True Negative, False Negative).

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

Line 112: `path=r'./PredictedData/'` **(Fill the path of the network result in step before generating contour line and heatmap)**

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
| Network Running Time (External Validation(Only Slide Data))  | 8.5837  |
| Network Running Time (External Validation(Only Tile Data))  | 36.2036  |
| Network Running Time (External Validation(Slide and Tile Data))  | 23.5408  |
| Post-Processing Time (External Validation(Only Slide Data))  | 0.4937  |
| Post-Processing Time (External Validation(Only Tile Data))  | 21.6656  |
| Post-Processing Time (External Validation(Slide and Tile Data))  | 9.4012  |
| Total Time (External Validation(Only Slide Data))  | 9.0774  |
| Total Time (External Validation(Only Tile Data))  | 57.8692  |
| Total Time (External Validation(Slide and Tile Data))  | 32.942  |
