#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import r2_score

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('Animal.csv')
#pd.read_csv("veriler.csv")

x = veriler.iloc[:,1:17].values #bağımsız değişkenler
y = veriler.iloc[:,17:].values #bağımlı değişken
print(y)

yumurta = veriler[['eggs']].values

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)

class_names = ['Tüy','Yüzgeç','Yumurta', 'Saç', 'Suda Yaşayan','Omurga','Bacak sayısı']


fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,colorbar=True,class_names=class_names)
plt.title("Sınıfı belirleyen özelliklerin biribiriyle ilişkisi")
plt.xlabel("Tahmin")
plt.ylabel("Doğru")
plt.show()

print("Karar Ağacı sınıflandırma doğruluk değeri:",metrics.accuracy_score(y_test,y_pred)," %")

print(r2_score(y_test, y_pred))