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

dataset = pd.read_csv('C:\\E-Shop.csv')

dataset.head()

X = final_data.drop('Transaction', axis = 1) # Features
Y = final_data['Transaction'] # Labels


feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)
smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)


rfc = RandomForestClassifier(n_estimators=300, criterion='entropy', max_features='auto')
#rfc.fit(X_train,Y_train)
#Y_pred = rfc.predict(X_test)
#conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
#plt.figure(figsize=(8,6))
#sns.heatmap(conf_mat,annot=True)
#plt.title("Confusion_matrix")
#plt.xlabel("Predicted Class")
#plt.ylabel("Actual class")
#plt.show()
#print('Confusion matrix: \n', conf_mat)
#print('TP: ', conf_mat[1,1])
#print('TN: ', conf_mat[0,0])
#print('FP: ', conf_mat[0,1])
#print('FN: ', conf_mat[1,0])