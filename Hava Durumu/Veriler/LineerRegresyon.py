# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:24:06 2019

@author: Casper
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.regression.linear_model as sm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

veri = pd.read_csv('Weather_predict.csv')
del veri['id']

veri2 = veri.apply(LabelEncoder().fit_transform)

x =veri2.iloc[:,0:9]
y =veri2.iloc[:,9:]
X =x.values
Y =y.values

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

r_ols1 = sm.OLS(endog = Y, exog =X)
r1 = r_ols1.fit()
print(r1.summary())

model1 = veri2.iloc[:,[0,5,6]].values

r_ols1 = sm.OLS(endog = Y, exog =model1)
r1 = r_ols1.fit()
print(r1.summary())

regressor.fit(x_train,y_train)

#Sınıflandırma
classifier = RandomForestClassifier(n_estimators = 400)

#Deneme
classifier.fit(x_train, y_train) 

#Tahmin yapma
tahmin = classifier.predict(x_test) 

print("Lineer Regresyon Doğruluk                           : ", accuracy_score(y_test, tahmin) *  100," %")

cm = confusion_matrix(y_test,tahmin)
sns.heatmap(cm,annot=True,fmt="d")
plt.savefig('conf_matrix.png')
plt.title('Lineer regresyon Confusion Matrix')
plt.show()
