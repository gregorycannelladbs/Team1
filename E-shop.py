# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:34:10 2020

@author: Greg
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from imblearn.over_sampling import SMOTE  # imblearn library can be installed in Visual Studio by going into Python Environment -> Install new package -> imblearn package
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Importing dataset and examining it
dataset = pd.read_csv("E-Shop.csv")
pd.set_option('display.max_columns', None) # Will ensure that all columns are displayed
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# Converting Categorical features into Numerical features
def converter(column):
    if column == 'TRUE':
        return 1
    else:
        return 0
    
def converter2(column):
    if column == 'Returning_Visitor':
        return 1
    else:
        return 0

dataset['Weekend'] = dataset['Weekend'].apply(converter)
dataset['VisitorType'] = dataset['VisitorType'].apply(converter2)

categorical_features = ['Month']
dataset = pd.get_dummies(dataset, columns = categorical_features)

print(dataset.info())
print(dataset.head(2))

# Dividing dataset into label and feature sets
X = dataset.drop('Transaction', axis = 1) # Feature Set (Just Indepedent Variables)
Y = dataset['Transaction'] # Label Column (Target Variable)
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)  

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Dividing dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

print(X_train.shape)
print(X_test.shape)

# Implementing Oversampling to balance the dataset; SMOTE stands for Synthetic Minority Oversampling TEchnique
print("Number of observations in each class before oversampling (training data): \n", pd.Series(Y_train).value_counts())

smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)

print("Number of observations in each class after oversampling (training data): \n", pd.Series(Y_train).value_counts())

# Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
rfc = RandomForestClassifier(criterion='entropy', max_features='auto', random_state=1)
grid_param = {'n_estimators': [50, 100, 150, 200, 250, 300]}

gd_sr = GridSearchCV(estimator=rfc, param_grid=grid_param, scoring='precision', cv=5)

"""
In the above GridSearchCV(), scoring parameter should be set as follows:
scoring = 'accuracy' when you want to maximize prediction accuracy
scoring = 'recall' when you want to minimize false negatives
scoring = 'precision' when you want to minimize false positives
scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
"""

gd_sr.fit(X_train, Y_train)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)

# Building random forest using the tuned parameter
rfc = RandomForestClassifier(n_estimators=50, criterion='entropy', max_features='auto', random_state=1)
rfc.fit(X_train,Y_train)
featimp = pd.Series(rfc.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimp)

Y_pred = rfc.predict(X_test)
conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])

# Selecting features with higher sifnificance and redefining feature set
X = dataset[['PageValue', 'ExitRate', 'ProductRelated_Duration', 'Administrative', 'ProductRelated']]

feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Dividing dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)

rfc = RandomForestClassifier(n_estimators=100, criterion='entropy', max_features='auto', random_state=1)
rfc.fit(X_train,Y_train)

Y_pred = rfc.predict(X_test)
conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])

##########################################################
#
## Implementing PCA to visualize dataset
#pca = PCA(n_components = 2)
#pca.fit(X_scaled)
#x_pca = pca.transform(X_scaled)
#print(pca.explained_variance_ratio_)
#print(sum(pca.explained_variance_ratio_))
#
#plt.figure(figsize = (8,6))
#plt.scatter(x_pca[:,0], x_pca[:,1], c=Y, cmap='plasma')
#plt.xlabel('First Principal Component')
#plt.ylabel('Second Principal Component')
#plt.show()
#
############################################################
#
## Implementing K-Means CLustering on dataset and visualizing clusters
#kmeans = KMeans(n_clusters = 2)
#kmeans.fit(X_scaled)
#plt.figure(figsize = (8,6))
#plt.scatter(x_pca[:,0], x_pca[:,1], c=kmeans.labels_, cmap='plasma')
#plt.xlabel('First Principal Component')
#plt.ylabel('Second Principal Component')
#plt.show()
#
## Finding the number of clusters (K)
#inertia = []
#for i in range(1,11):
#    kmeans = KMeans(n_clusters = i, random_state = 100)
#    kmeans.fit(X_scaled)
#    inertia.append(kmeans.inertia_)
#
#plt.plot(range(1, 11), inertia)
#plt.title('The Elbow Plot')
#plt.xlabel('Number of clusters')
#plt.ylabel('Inertia')
#plt.show()
