# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 16:28:28 2020

@author: Nick
"""

import pandas as pd
import sklearn
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier

labels = ['buying','maint','doors','persons','lug_boot','safety','class']
df = pd.read_csv("car.data", sep = "," , names = labels)

x = np.array(df.drop('class',1))
le = preprocessing.LabelEncoder()
y = le.fit_transform(np.array(df['class']))
for i in range(6):
    x[:,i] = le.fit_transform(x[:,i])
    
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1)

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(x_train, y_train)
r2 = knn.score(x_test, y_test)
print (r2)

predict = knn.predict(x_test)
output_labels = ['unacc','acc','good','vgood']

for i in range(len(predict)):
    print("Car",i,"Stats:",x_test[i,:],"| Prediction:",output_labels[predict[i]],"| Actual Rating:",output_labels[y_test[i]])