# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:19:53 2019

@author: Casper
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.regression.linear_model as sm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

veri = pd.read_csv('Weather_predict.csv')
del veri['id']

veri2 = veri.apply(LabelEncoder().fit_transform)

x =veri2.iloc[:,0:9]
y =veri2.iloc[:,9:]
X =x.values
Y =y.values

print("-----------------Linear Regresyon----------------------------")

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

r_ols1 = sm.OLS(endog = Y, exog =X)
r1 = r_ols1.fit()
print(r1.summary())
tahmin1 = regressor.predict(x_test)

print("-----------------Rassal Ağaç modeli tahmini----------------------------")
rf_reg = RandomForestRegressor(n_estimators = 8, random_state=1)
rf_reg.fit(X,Y)
model2 = sm.OLS(rf_reg.predict(X),X)
print(model2.fit().summary())

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)
tahmin2 = rf_reg.predict(x_test)

cm = confusion_matrix(y_test,tahmin2.round())

print("Confisuon Matrix: \n",cm)
print("Sınıflandırma raporu: \n",classification_report(y_test,tahmin2.round()))

sns.heatmap(cm,annot=True,fmt="d")
plt.savefig('conf_matrix.png')
plt.title('Rassal ağaça Confusion Matrix')
plt.show()

print("-----------------Knn modeli tahmini----------------------------")
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)
knn = KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(x_train,y_train)
print('Knn OLS')
model3 = sm.OLS(knn.predict(X),X)
print(model3.fit().summary())

tahmin3 = knn.predict(x_test)

print("-----------------Karar Ağacı modeli tahmini----------------------------")
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x_train,y_train)
model4 = sm.OLS(r_dt.predict(x_train),x_train)
print(model4.fit().summary())

tahmin4 = r_dt.predict(x_test)

print("---------------Logistic Regresyon Model Tahmini------------------------")
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

tahmin5 = logr.predict(X_test)


print("-------------------Algoritmaların doğruluk değerleri-------------------------")

#Model performansı
print("Lineer Regresyon Doğruluk                           : ", accuracy_score(y_test, tahmin1.round()) *  100," %")

print("Rassal ağaç regresyon modeli doğruluk               : ", accuracy_score(y_test, tahmin2.round()) *  100," %")    

print("KNN regresyon modeli doğruluk                       : ", accuracy_score(y_test, tahmin3.round()) *  100," %") 

print("Karar Ağacı regresyon modeli doğruluk               : ", accuracy_score(y_test, tahmin4.round()) *  100," %") 

print("Logistic regresyon modeli doğruluk                  : ", accuracy_score(y_test, tahmin5.round()) *  100," %") 



print("------------------Algortimaların R2 değerleri--------------------------")

print("Linear Regresyon R2 degeri:")
print(r2_score(y_test,tahmin1,multioutput='variance_weighted'))


print("Rassal Ormana Regresyon R2 degeri:")
print(r2_score(y_test,tahmin2,multioutput='variance_weighted'))

print("Knn R2 degeri:")
print(r2_score(y_test,tahmin3,multioutput='variance_weighted'))


print("Karar Ağacı R2 degeri:")
print(r2_score(y_test,tahmin4,multioutput='variance_weighted'))

print("Logistic Regresyon R2 degeri:")
print(r2_score(y_test,tahmin5,multioutput='variance_weighted'))
