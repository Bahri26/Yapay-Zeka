# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Ders 1:Kütüphanelerin yüklenmesi
#Kütüphaneler 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
import statsmodels.regression.linear_model as sm
from sklearn.preprocessing import LabelEncoder

#Veri yükleme
veri = pd.read_csv("Animal.csv")

x = veri.iloc[:,1:17]
y = veri.iloc[:,17]
X = x.values
Y = y.values

dogum   = veri[['airborne']]
yumurta = veri[['eggs']]

model1 = veri.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]].values

r_ols1 = sm.OLS(endog = Y, exog = model1)
r1 = r_ols1.fit()
print(r1.summary())

x_train, x_test, y_train, y_test = train_test_split(model1,Y, test_size=0.33, random_state=1)

clf = DecisionTreeClassifier()

clf = clf.fit(x_train,y_train)

tahmin = clf.predict(x_test)

print("Accuracy1:",metrics.accuracy_score(y_test, tahmin))

model1 = veri.iloc[:,[2,3,4,6,8,9,10,11,16]].values

r_ols1 = sm.OLS(endog = Y, exog = model1)
r1 = r_ols1.fit()
print(r1.summary())

x_train, x_test, y_train, y_test = train_test_split(model1,Y, test_size=0.33, random_state=1)

clf = DecisionTreeClassifier()

clf = clf.fit(x_train,y_train)

tahmin = clf.predict(x_test)

print("Accuracy1:",metrics.accuracy_score(y_test, tahmin))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, tahmin)
print(cm)

x_train, x_test, y_train, y_test = train_test_split(dogum,y, test_size=0.33, random_state=1)

clf = DecisionTreeClassifier()

clf = clf.fit(x_train,y_train)
x_train = x_train.sort_index()
y_train = y_train.sort_index()

fig, ax = plt.subplots()
plt.plot(x_train,y_train)
plt.plot(x_test,tahmin)

plt.title("Hayvanların sınıflandırılması")
plt.xlabel("Doğum")
plt.ylabel("Sınıflar")




