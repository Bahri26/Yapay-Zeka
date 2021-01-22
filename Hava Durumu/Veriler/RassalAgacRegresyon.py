# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 17:09:11 2019

@author: Casper
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import statsmodels.regression.linear_model as sm
from sklearn.metrics import confusion_matrix
import seaborn as sns

veri = pd.read_csv('Weather_predict.csv')
del veri['id']

veri2 = veri.apply(LabelEncoder().fit_transform)

x =veri2.iloc[:,0:9]
y =veri2.iloc[:,9:]
X =x.values
Y =y.values

nem = veri2[['Mouisture']].reshape(1,-1)

print("-----------------Rassal Ağaç modeli tahmini----------------------------")
rf_reg = RandomForestRegressor(n_estimators = 8, random_state=1)
rf_reg.fit(X,Y)
model2 = sm.OLS(rf_reg.predict(X),X)
print(model2.fit().summary())

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)
tahmin = rf_reg.predict(x_test)

cm = confusion_matrix(y_test,tahmin.round())
sns.heatmap(cm,annot=True,fmt="d")
plt.savefig('conf_matrix.png')
plt.title('Rassaal ağaç regresyon algoritması Confusion Matrix')
plt.show()

x_train, x_test,y_train,y_test = train_test_split(nem,Y,test_size=0.33, random_state=0)
rf_reg.fit(x_train,y_train)
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,tahmin)
plt.title("Yağışı etkileyen faktörler")
plt.xlabel("Nem")
plt.ylabel("Yağış")
plt.savefig('yağış.png')
plt.show()

