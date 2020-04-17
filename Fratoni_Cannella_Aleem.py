# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:15:51 2020

@author: shakir
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # imblearn library can be installed using pip install imblearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

dataset = pd.read_csv('E-Shop.csv')
dataset = dataset.drop(['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration'], axis = 1)

print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# def converterVisitor(column):
#     if column == 'Returning_Visitor':
#         return 1
#     else:
#         return 0

# def converterWeekend(column):
#     #print(column)
#     if str(column) == 'True':
#         return 1
#     else:
#         return 0
#dataset['Transaction'] = dataset['Transaction'].apply(converterWeekend)
#dataset['VisitorType'] = dataset['VisitorType'].apply(converterVisitor)
#dataset['Weekend'] = dataset['Weekend'].apply(converterWeekend)

categorical_features = ['Month','VisitorType']
final_data = pd.get_dummies(dataset, columns = categorical_features)
print(final_data.info())
print(final_data.head(2))

X = final_data.drop('Transaction', axis = 1) # Features
Y = final_data['Transaction'] # Labels
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)

feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

print(X_train.shape)
print(X_test.shape)

# Implementing Oversampling to balance the dataset; SMOTE stands for Synthetic Minority Oversampling TEchnique
print("Number of observations in each class before oversampling (training data): \n", pd.Series(Y_train).value_counts())

smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)

print("Number of observations in each class after oversampling (training data): \n", pd.Series(Y_train).value_counts())

rfc = RandomForestClassifier(criterion='entropy', max_features='auto',random_state=1)
grid_param = {'n_estimators': [1,50, 100, 150, 200, 250, 300]}

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
rfc = RandomForestClassifier(n_estimators=1, criterion='entropy', max_features='auto', random_state=1)
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

## Selecting features with higher sifnificance and redefining feature set
X = final_data[['PageValue', 'ExitRate', 'ProductRelated_Duration', 'ProductRelated', 'BounceRate', 'Month_Nov', 'Month_May','Month_Mar','Weekend']]

feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Dividing dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)

rfc = RandomForestClassifier(n_estimators=1, criterion='entropy', max_features='auto', random_state=1)
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