#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import math
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from mpl_toolkits.mplot3d import Axes3D  # 3d Plot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import random

def stat_output(name, Y_pred, conf_matrix):

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
    if test_idx:
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


Train_data_path="801010/"
Result_path=""

os.makedirs(Result_path,exist_ok=True)

data = pd.read_csv(Train_data_path+'train_cross10.csv')
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


data2 = pd.read_csv(Train_data_path+'test_cross10.csv')
dataset2 = data2
presplitted2 = 0

Id2 = data2['id']

Id2 = list(Id2)
Id_n2 = []
for s in Id2:
    s = s[:s.rfind('-')]
    Id_n2.append(s)
Id_n2 = list(set(Id_n2))

random.shuffle(Id_n2, random.random)
Id_test = Id_n2

test_df = data2.iloc[0:0, :]

train_df = data.iloc[0:0, :]


for a in Id_test:

    s =a + '-1'
    li = [s]
    temp = data2.loc[data2.id.isin(li)]
    test_df = test_df.append(temp, ignore_index=True)

    s = a + '-2'
    li = [s]
    temp = data2.loc[data2.id.isin(li)]
    test_df = test_df.append(temp, ignore_index=True)

    s = a + '-3'
    li = [s]
    temp = data2.loc[data2.id.isin(li)]
    test_df = test_df.append(temp, ignore_index=True)

    try:
        s = a + '-4'
        li = [s]
        temp = data2.loc[data2.id.isin(li)]
        test_df = test_df.append(temp, ignore_index=True)

        s = a + '-5'
        li = [s]
        temp = data2.loc[data2.id.isin(li)]
        test_df = test_df.append(temp, ignore_index=True)

        s = a + '-6'
        li = [s]
        temp = data2.loc[data2.id.isin(li)]
        test_df = test_df.append(temp, ignore_index=True)

        s = a + '-7'
        li = [s]
        temp = data2.loc[data2.id.isin(li)]
        test_df = test_df.append(temp, ignore_index=True)

        s = a + '-8'
        li = [s]
        temp = data2.loc[data2.id.isin(li)]
        test_df = test_df.append(temp, ignore_index=True)
    except:
        pass

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
    try:
        s = a + '-4'
        li = [s]
        temp = data2.loc[data2.id.isin(li)]
        test_df = test_df.append(temp, ignore_index=True)

        s = a + '-5'
        li = [s]
        temp = data2.loc[data2.id.isin(li)]
        test_df = test_df.append(temp, ignore_index=True)

        s = a + '-6'
        li = [s]
        temp = data2.loc[data2.id.isin(li)]
        test_df = test_df.append(temp, ignore_index=True)

        s = a + '-7'
        li = [s]
        temp = data2.loc[data2.id.isin(li)]
        test_df = test_df.append(temp, ignore_index=True)

        s = a + '-8'
        li = [s]
        temp = data2.loc[data2.id.isin(li)]
        test_df = test_df.append(temp, ignore_index=True)
    except:
        pass

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


#Logistic regression --lbfgs

name = "Logistic regression --lbfgs"

classifier = LogisticRegression(solver='lbfgs', random_state=rn)

classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"logistic_regression_lbfgs.csv")
intercept = classifier.intercept_
column_names = dataset.columns
column_names = column_names.drop(['id'])
# column_names = column_names.drop(['cell no.'])
column_names = column_names.drop(['Import_diagnosis'])
coef = pd.DataFrame(classifier.coef_.transpose(),
                    index=column_names,
                    columns=['Coefficients'])
print('Intercept = %f\n' % intercept)
inverse_intercept = math.exp(intercept) / (1 + math.exp(intercept))
print('baseline probability of Cancerous diagnosis by inverse intercept = %f\n' % inverse_intercept)
print(coef)
cm = confusion_matrix(Y_test, Y_pred)
cmLog = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)



#Logistic regression --sag

name = "Logistic regression --sag"

classifier = LogisticRegression(solver='sag', random_state=rn)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"logistic_regression_sag.csv")
intercept = classifier.intercept_
column_names = dataset.columns
column_names = column_names.drop(['id'])
# column_names = column_names.drop(['cell no.'])
column_names = column_names.drop(['Import_diagnosis'])
coef = pd.DataFrame(classifier.coef_.transpose(),
                    index=column_names,
                    columns=['Coefficients'])
print('Intercept = %f\n' % intercept)
inverse_intercept = math.exp(intercept) / (1 + math.exp(intercept))
print('baseline probability of Cancerous diagnosis by inverse intercept = %f\n' % inverse_intercept)
print(coef)
cm = confusion_matrix(Y_test, Y_pred)
cmLog = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)



#Logistic regression --saga

name = "Logistic regression --saga"

classifier = LogisticRegression(solver='saga', random_state=rn)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"logistic_regression_saga.csv")
intercept = classifier.intercept_
column_names = dataset.columns
column_names = column_names.drop(['id'])
# column_names = column_names.drop(['cell no.'])
column_names = column_names.drop(['Import_diagnosis'])
coef = pd.DataFrame(classifier.coef_.transpose(),
                    index=column_names,
                    columns=['Coefficients'])
print('Intercept = %f\n' % intercept)
inverse_intercept = math.exp(intercept) / (1 + math.exp(intercept))
print('baseline probability of Cancerous diagnosis by inverse intercept = %f\n' % inverse_intercept)
print(coef)
cm = confusion_matrix(Y_test, Y_pred)
cmLog = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)



#Logistic regression --liblinear

name = "Logistic regression --liblinear"

classifier = LogisticRegression(solver='liblinear', random_state=rn)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"logistic_regression_liblinear.csv")
intercept = classifier.intercept_
column_names = dataset.columns
column_names = column_names.drop(['id'])
# column_names = column_names.drop(['cell no.'])
column_names = column_names.drop(['Import_diagnosis'])
coef = pd.DataFrame(classifier.coef_.transpose(),
                    index=column_names,
                    columns=['Coefficients'])
print('Intercept = %f\n' % intercept)
inverse_intercept = math.exp(intercept) / (1 + math.exp(intercept))
print('baseline probability of Cancerous diagnosis by inverse intercept = %f\n' % inverse_intercept)
print(coef)
cm = confusion_matrix(Y_test, Y_pred)
cmLog = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)





#Logistic regression --newton-cg

name = "Logistic regression --newton-cg"

classifier = LogisticRegression(solver='newton-cg', random_state=rn)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"logistic_regression_newton_cg.csv")

intercept = classifier.intercept_
column_names = dataset.columns
column_names = column_names.drop(['id'])
# column_names = column_names.drop(['cell no.'])
column_names = column_names.drop(['Import_diagnosis'])
coef = pd.DataFrame(classifier.coef_.transpose(),
                    index=column_names,
                    columns=['Coefficients'])
print('Intercept = %f\n' % intercept)
inverse_intercept = math.exp(intercept) / (1 + math.exp(intercept))
print('baseline probability of Cancerous diagnosis by inverse intercept = %f\n' % inverse_intercept)
print(coef)
cm = confusion_matrix(Y_test, Y_pred)
cmLog = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)



#K-NN Algorithm, manhattan

name = "K-NN Algorithm (n_neighbors = 1, metric = 'manhattan', p = 7)"
classifier = KNeighborsClassifier(n_neighbors=1, metric='manhattan', p=7)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"K-NN Algorithm (n_neighbors = 1, metric = 'manhattan', p = 7).csv")
cm = confusion_matrix(Y_test, Y_pred)
cmKNN = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)



#K-NN Algorithm, chebyshev

name = "K-NN Algorithm (n_neighbors = 3, metric = 'chebyshev', p = 2)"
classifier = KNeighborsClassifier(n_neighbors=3, metric='chebyshev', p=2)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"K-NN Algorithm (n_neighbors = 3, metric = 'chebyshev', p = 2).csv")
cm = confusion_matrix(Y_test, Y_pred)
cmKNN = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)



#K-NN Algorithm, manhattan

name = "K-NN Algorithm (n_neighbors = 3, metric = 'manhattan', p = 2)"
classifier = KNeighborsClassifier(n_neighbors=3, metric='manhattan', p=2)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"K-NN Algorithm (n_neighbors = 3, metric = 'manhattan', p = 2).csv")
cm = confusion_matrix(Y_test, Y_pred)
cmKNN = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)



#K-NN Algorithm, minkowski

name = "K-NN Algorithm (n_neighbors = 3, metric = 'minkowski', p = 2)"
classifier = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"K-NN Algorithm (n_neighbors = 3, metric = 'minkowski', p = 2).csv")

cm = confusion_matrix(Y_test, Y_pred)
cmKNN = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)

# try changing k
print('Minimization of misclassification error by trying different numer of neighbours:')
# creating list of K for KNN
k_list = list(range(1, 20))
# creating list of cv scores
cv_scores = []

# perform 10-fold cross validation
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
MSE = [1 - x for x in cv_scores]

# finding best k
best_k = k_list[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d." % best_k)


print('--- Minkowski distance metric 1 to 10---')
k_list = list(range(1, 10))
cv_scores = []
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=best_k, metric='minkowski', p=k)
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    name = "K-NN Algorithm (n_neighbors = {}, metric = 'minkowski', p = {})".format(best_k, k)
MSE = [1 - x for x in cv_scores]
best_p = k_list[MSE.index(min(MSE))]
print("The optimal number of p is %d." % best_p)

print('--- Uniform weight or distance weight?---')
knn = KNeighborsClassifier(n_neighbors=best_k, weights='uniform', metric='minkowski', p=best_p)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"K-NN Algorithm (n_neighbors= best_k, weights='uniform', metric = 'minkowski', p = best_p).csv")

acc = accuracy_score(Y_test, Y_pred)
print('Accuracy score under uniform weight = {}'.format(acc))

knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='minkowski', p=best_p)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"K-NN Algorithm (n_neighbors= best_k, weights='distance', metric = 'minkowski', p = best_p).csv")

acc = accuracy_score(Y_test, Y_pred)
print('Accuracy score under distance weight = {}'.format(acc))




# SVM--linear

name = "SVM--linear"
classifier = SVC(kernel='linear', gamma='scale', random_state=rn)
classifier.fit(X_train, Y_train)
for (intercept, coef) in zip(classifier.intercept_, classifier.coef_):
    s = "y = {0:.3f}".format(intercept)
    for (i, c) in enumerate(coef):
        s += " + {0:.3f} * x{1}".format(c, i)
print("decision boundary: " + s)
coef = pd.DataFrame(classifier.coef_.transpose(),
                    index=column_names,
                    columns=['Coefficients'])
print(coef)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"SVC(kernel = 'linear', gamma='scale', random_state = rn).csv")

cm = confusion_matrix(Y_test, Y_pred)
cmSVC = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)



# SVM--poly

name = "SVM--poly"
classifier = SVC(kernel='poly', gamma='scale', random_state=rn)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"SVC(kernel = 'poly', gamma='scale', random_state = rn).csv")

cm = confusion_matrix(Y_test, Y_pred)
cmSVC = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)




# SVM--rbf

name = "SVM--rbf"
classifier = SVC(kernel='rbf', gamma='scale', random_state=rn)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"SVC(kernel = 'rbf', gamma='scale', random_state = rn).csv")

cm = confusion_matrix(Y_test, Y_pred)
cmSVC = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)


# SVM--sigmoid

name = "SVM--sigmoid"
classifier = SVC(kernel='sigmoid', gamma='scale', random_state=rn)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"SVC(kernel = 'sigmoid', gamma='scale', random_state = rn).csv")

cm = confusion_matrix(Y_test, Y_pred)
cmSVC = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)



# K-SVM

name = "K-SVM"
classifier = SVC(kernel='rbf', gamma='scale', random_state=rn)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"K-SVM.csv")
cm = confusion_matrix(Y_test, Y_pred)
cmKSVM = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)



print('--- Naive Bayes classifier---')



# Naive_Bayes

name = "Naive_Bayes"
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"Naive_Bayes.csv")

cm = confusion_matrix(Y_test, Y_pred)
cmBayes = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)



# Fitting Decision Tree Algorithm --entropy

name = "Fitting Decision Tree Algorithm --entropy"
classifier = DecisionTreeClassifier(criterion='entropy', random_state=rn)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"DecisionTreeClassifier(criterion = 'entropy', random_state = rn).csv")

cm = confusion_matrix(Y_test, Y_pred)
cmDecTree = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)



# Fitting Decision Tree Algorithm --gini

name = "Fitting Decision Tree Algorithm --gini"
classifier = DecisionTreeClassifier(criterion='gini', random_state=rn)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"ecisionTreeClassifier(criterion = 'gini', random_state = rn).csv")

cm = confusion_matrix(Y_test, Y_pred)
cmDecTree = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)




# Random Forest Classification Algorithm -10entropy

name = "Random Forest Classification Algorithm -10entropy"
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=rn)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = rn).csv")

cm = confusion_matrix(Y_test, Y_pred)
cmRanForest = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)



# Random Forest Classification Algorithm -10gini

name = "Random Forest Classification Algorithm -10gini"
classifier = RandomForestClassifier(n_estimators=10, criterion='gini', random_state=rn)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = rn).csv")

cm = confusion_matrix(Y_test, Y_pred)
cmRanForest = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)



# Random Forest Classification Algorithm -100entropy

name = "Random Forest Classification Algorithm -100entropy"
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=rn)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = rn).csv")

cm = confusion_matrix(Y_test, Y_pred)
cmRanForest = cm
out = out.append(stat_output(name, Y_pred, cm), ignore_index=True)





# Random Forest Classification Algorithm -100gini

name = "Random Forest Classification Algorithm -100gini"
classifier = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=rn)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

output = []
for i in range(len(Y_pred)):
    output.append([])
    output[i].append(x_testid[i])
    output[i].append(Y_test[i])
    output[i].append(Y_pred[i])

output_pd = pd.DataFrame(columns=output_properties, data=output)
output_pd.to_csv(Result_path+"RandomForestClassifier(n_estimators = 100, criterion = 'gini', random_state = rn).csv")




