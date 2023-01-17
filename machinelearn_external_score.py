#!/usr/bin/env python
# coding: utf-8


import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from mpl_toolkits.mplot3d import Axes3D  # 3d Plot
import random

thershold=.005

def stat_output(name, Y_pred, conf_matrix):
    from sklearn.metrics import f1_score
    from sklearn.model_selection import cross_val_score
    conf_matrix_sum = conf_matrix.sum()
    c = (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix_sum
    CrossValScores = cross_val_score(classifier, X_train, Y_train, cv=10)
    print('Prediction:')
    print(Y_pred)
    print("{} accuracy by confusion matrix: {}".format(name, c))
    print("Accuracy by cross validation: %0.2f (+/- %0.2f)" % (CrossValScores.mean(), CrossValScores.std() * 2))
    print("conf_matrix =")
    print(conf_matrix)

    TN = conf_matrix[0][0]  # True negatives
    TP = conf_matrix[1][1]  # True positives
    FN = conf_matrix[1][0]  # False negatives
    FP = conf_matrix[0][1]  # False positives

    # ---------------------------------------------------------------
    # Compute Ratios
    # ---------------------------------------------------------------

    TPR = float(TP) / (TP + FN)
    print('TPR (a.k.a. recall) = %4.2f%%' % (TPR * 100))

    TNR = float(TN) / (TN + FP)
    print('TNR = %4.2f%%' % (TNR * 100))

    PPV = float(TP) / (TP + FP)
    print('PPV = %4.2f%%' % (PPV * 100))

    NPV = float(TN) / (TN + FN)
    print('NPV = %4.2f%%' % (NPV * 100))

    # use formula
    print("f1 score (testing, predict) = {}".format(f1_score(Y_test, Y_pred, average=None)))

    Accuracy = (float(TN) + float(TP)) / (TN + TP + FP + FN)
    listPara = [name, TPR, TNR, PPV, NPV, Accuracy]
    seriesPara = pd.Series(listPara, index=columns)
    return (seriesPara)


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, title, test_idx=None, resolution=0.02):
    from matplotlib.colors import ListedColormap
    import warnings
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('blue', 'red', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.title(title, fontsize=16, fontweight='bold')
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
import random

result_path="external/"

data = pd.read_csv('trainingdata.csv')
dataset = data
presplitted = 0

Id = data['id']

Id = list(Id)
Id_n = []
for s in Id:
    s = s[:s.rfind('-')]
    Id_n.append(s)
Id_n = list(set(Id_n))



random.shuffle(Id_n, random.random)
Id_train = Id_n


data2 = pd.read_csv('data/External_SlideData_gt.csv')
dataset2 = data2
presplitted2 = 0

Id2 = data2['id']

Id2 = list(Id2)
Id_n2 = []
for s in Id2:
    s = s[:s.rfind(',')]
    Id_n2.append(s)
Id_n2 = list(set(Id_n2))


random.shuffle(Id_n2, random.random)
Id_test = Id_n2


print("Test Len: ", len(Id_test))
print("Train Len: ", len(Id_train))

test_df = data2.iloc[0:0, :]

train_df = data.iloc[0:0, :]


for a in Id_test:

    s = a + ', 1].txt'
    li = [s]
    temp = data2.loc[data2.id.isin(li)]
    test_df = test_df.append(temp, ignore_index=True)

    s = a + ', 2].txt'
    li = [s]
    temp = data2.loc[data2.id.isin(li)]
    test_df = test_df.append(temp, ignore_index=True)

    s = a + ', 3].txt'
    li = [s]
    temp = data2.loc[data2.id.isin(li)]
    test_df = test_df.append(temp, ignore_index=True)

for a in Id_train:
    s = a + '-1'
    li = [s]
    temp = data.loc[data.id.isin(li)]
    train_df = train_df.append(temp, ignore_index=True)

    s = a + '-2'
    li = [s]
    temp = data.loc[data.id.isin(li)]
    train_df = train_df.append(temp, ignore_index=True)

    s = a + '-3'
    li = [s]
    temp = data.loc[data.id.isin(li)]
    train_df = train_df.append(temp, ignore_index=True)

test_df = test_df.reindex(np.random.permutation(data2.index))
train_df = train_df.reindex(np.random.permutation(data.index))

test_df = test_df.dropna()
train_df = train_df.dropna()

x_trainid = train_df.iloc[:, 0].values
X_train = train_df.iloc[:, 1:42].values
Y_train = train_df.iloc[:, 42].values


x_testid = test_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:42].values
Y_test = test_df.iloc[:, 42].values


labelencoder_Y = LabelEncoder()
Y_train = labelencoder_Y.fit_transform(Y_train)
Y_test = labelencoder_Y.fit_transform(Y_test)


labelencoder_Y_name_mapping = dict(zip(labelencoder_Y.classes_, labelencoder_Y.transform(labelencoder_Y.classes_)))
print("encoding result: {}".format(labelencoder_Y_name_mapping))



sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

output_properties = ["id", "gt diag", "pred diag"]


pca = PCA(n_components='mle')
X_train_reduced = pca.fit_transform(X_train)
print(
    "TEST: data dimension reduced by PCA. mle suggest data dimension = {}, total explained variance ratio = {}".format(
        X_train_reduced.shape[1], np.sum(pca.explained_variance_ratio_)))
pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train)
print("To plot decision boundaries, PCA(n_components = 2) is used. Total explained variance ratio = {}".format(
    np.sum(pca.explained_variance_ratio_)))
X_test_reduced = pca.fit_transform(X_test)

# PCA for 3d
pca = PCA(n_components=3)
X_train_reduced_3 = pca.fit_transform(X_train)
print("To plot decision boundaries, PCA(n_components = 2) is used. Total explained variance ratio = {}".format(
    np.sum(pca.explained_variance_ratio_)))
X_test_reduced_3 = pca.fit_transform(X_test)






columns = ['name', 'TPR', 'TNR', 'PPV', 'NPV', 'Accuracy']
out = pd.DataFrame(columns=columns)

rn = random.randrange(101)



# Loop

name_group=["Logistic regression --lbfgs","Logistic regression --sag","Logistic regression --saga","Logistic regression --liblinear","Logistic regression --newton-cg","SVM--linear"]
classifier_group=[LogisticRegression(solver='lbfgs', random_state=rn),LogisticRegression(solver='sag', random_state=rn),LogisticRegression(solver='saga', random_state=rn),LogisticRegression(solver='liblinear', random_state=rn),LogisticRegression(solver='newton-cg', random_state=rn),SVC(kernel = 'linear', gamma='scale', random_state = rn)]

for i in range(6):
    name = name_group[i]
    classifier = classifier_group[i]

    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    output = []
    for i in range(len(Y_pred)):
        output.append([])
        output[i].append(x_testid[i])
        output[i].append(Y_test[i])
        output[i].append(Y_pred[i])

    output_pd = pd.DataFrame(columns=output_properties, data=output)

    data = pd.read_csv('data/External_SlideData.csv')
    data.info()
    ref_data = pd.read_csv("data/GroundTruth_External.csv")

    X = data.iloc[:, 1:42].values
    Id = data.iloc[:, 0].values

    X = sc.transform(X)
    coef = classifier.coef_[0]
    intercept = classifier.intercept_
    Y = []
    for x in X:
        score = 0
        for i in range(len(x)):
            score += x[i] * coef[i]
        score += intercept
        Y += [1 - (1 / (math.exp(-score) + 1))]

    result = pd.DataFrame({'id': Id, 'pre_probi': Y})
    columns = ['ID', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'Score', 'PredictDiagnosis']
    report = pd.DataFrame(columns=columns)
    for index, row in result.iterrows():
        idStr = row[0][::-1]
        tempStr = ''
        remove = False
        if '-' in idStr:
            for l in idStr:
                if remove:
                    tempStr = tempStr + l
                if l == '-':
                    remove = True
        else:
            for l in idStr:
                if remove:
                    tempStr = tempStr + l
                if l == '.':
                    remove = True
        result.loc[index, 'id'] = tempStr[::-1]
    ID = result['id']
    for c in ID.unique():
        s = result.loc[result['id'] == c, 'pre_probi']
        pre = False

        pdct = 0

        for per in s:
            if per >= .5:
                pre = True

        series = [c]
        series = series + list(s)

        if len(series) != 7:
            for count in range(7 - len(series)):
                series = series + ['NA']

        if len(series) == 7:
            series = series + [float(max(s))] + [pre]
            ap = pd.Series(series, index=columns)
        report = report.append(ap, ignore_index=True)
    report.drop(["S1", "S2", "S3", "S4", "S5", "S6"], inplace=True, axis=1)

    report_neg = report.loc[report['PredictDiagnosis'] == False]
    report_pos = report.loc[report['PredictDiagnosis'] == True]

    os.makedirs(result_path + name,exist_ok=True)

    posName = result_path + name + '/' + 'report_pos_' + '.csv'
    negName = result_path + name + '/' + 'requireTile_' + '.csv'

    report_pos.sort_values(by='Score', ascending=False).to_csv(posName, index=False)
    report_neg.to_csv(negName, index=False)
