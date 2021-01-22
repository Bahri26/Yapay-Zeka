# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 18:53:48 2019

@author: Casper
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import statsmodels.regression.linear_model as sm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

veri = pd.read_csv("Animal.csv")

x = veri.iloc[:,1:17]
y = veri.iloc[:,17]
X = x.values
Y = y.values

dogum   = veri[['airborne']]

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

knn = KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(x_train,y_train)
tahmin = knn.predict(x_test)
print(metrics.accuracy_score(y_test,tahmin))

model = veri.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]].values

r_ols1 = sm.OLS(endog = Y, exog = model)
r1 = r_ols1.fit()
print(r1.summary())

model = veri.iloc[:,[2,3,8,11,12,13]].values

r_ols1 = sm.OLS(endog = Y, exog = model)
r1 = r_ols1.fit()
print(r1.summary())

x_train, x_test,y_train,y_test = train_test_split(model,Y,test_size=0.33, random_state=0)
knn.fit(x_train,y_train)
tahmin = knn.predict(x_test)
print(metrics.accuracy_score(y_test,tahmin))

x_train, x_test,y_train,y_test = train_test_split(dogum,y,test_size=0.33, random_state=0)
knn.fit(x_train,y_train)
x_train = x_train.sort_index()
y_train = y_train.sort_index()

fig, ax = plt.subplots()
plt.plot(x_train,y_train)
plt.plot(x_test,tahmin)

plt.title("Hayvanların sınıflandırılması")
plt.xlabel("Doğum")
plt.ylabel("Sınıflar")
