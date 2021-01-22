#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve
#2. Veri Onisleme

#2.1. Veri Yukleme
Animal = pd.read_csv('Animal.csv')
Class = pd.read_csv('class.csv')
#pd.read_csv("veriler.csv")

Animal.head()
Animal.info()
Animal.describe()
Animal.drop("animal_name",axis=1,inplace=True)

color_list = [("red" if i ==1 else "blue" if i ==0 else "yellow" ) for i in Animal.hair]

unique_list = list(set(color_list))
unique_list

sns.countplot(x="hair", data=Animal)
plt.xlabel("Saç")
plt.ylabel("Sayı")
plt.show()
Animal.loc[:,'hair'].value_counts()


x = Animal.iloc[:,1:17].values #bağımsız değişkenler
y = Animal.iloc[:,17:].values #bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
x,y = Animal.loc[:,Animal.columns != 'hair'], Animal.loc[:,'hair']
knn.fit(x,y)
tahmin = knn.predict(x)
print("Tahmin = ",tahmin)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 1)
x,y = Animal.loc[:,Animal.columns != 'hair'], Animal.loc[:,'hair']
knn.fit(x_train,y_train)
tahmin = knn.predict(x_test)
print('Knn ile  (K=1) doğruluktur: ',knn.score(x_test,y_test))

k_values = np.arange(1,25)
train_accuracy = []
test_accuracy = []

for i, k in enumerate(k_values):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))

plt.figure(figsize=[13,8])
plt.plot(k_values, test_accuracy, label =  'Test Değeri')
plt.plot(k_values, train_accuracy, label = 'Deneme Değeri')
plt.legend()
plt.title('Değer ve Doğruluk')
plt.xlabel('Komşuluk sayıları')
plt.ylabel('Doğruluk')
plt.xticks(k_values)
plt.savefig('Eniyikomsuluk.png')
plt.show()
print("En iyi doğrulukdur {} ile K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

x = np.array(Animal.loc[:,"eggs"]).reshape(-1,1)
y = np.array(Animal.loc[:,'hair']).reshape(-1,1)

plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel('Yumurta')
plt.ylabel('Saç')
plt.savefig('Yumurta-sac.png')
plt.show()


regression = LinearRegression()
k=5
cv_result = cross_val_score(regression,x,y,cv=k)
print("CV Scores: ",cv_result)
print("CV Average: ",np.sum(cv_result)/k)

predict_space = np.linspace(min(x),max(x)).reshape(-1,1)
regression.fit(x,y)
predicted = regression.predict(predict_space)

print("R^2 Score: ",regression.score(x,y))

plt.plot(predict_space, predicted, color='black', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('Egg')
plt.ylabel('Milk')
plt.show()

x,y = Animal.loc[:,(Animal.columns != 'hair')], Animal.loc[:,'hair']
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.3, random_state=42)
knn.fit(x_train,y_train)
y_pred_prob = knn.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('Yanlış pozitif oran')
plt.ylabel('Doğru pozitif oran')
plt.title('ROC-eğrisi')
plt.savefig('ROC-eğrisi.png')
plt.show()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 1)
x,y = Animal.loc[:,Animal.columns != 'hair'], Animal.loc[:,'hair']
knn.fit(x_train,y_train)
tahmin = knn.predict(x_test)
print('Knn ile  (K=1) doğruluktur: ',knn.score(x_test,y_test))

cm = confusion_matrix(y_test,tahmin)

print("Confisuon Matrix: \n",cm)
print("Sınıflandırma raporu: \n",classification_report(y_test,tahmin))

sns.heatmap(cm,annot=True,fmt="d")
plt.savefig('conf_matrix.png')
plt.title('knn algoritması Confusion Matrix')
plt.show()

sns.countplot(x="hair", data=Animal)
plt.title('Hayvanlardaki saç sayısı')
plt.xlabel("Saç")
plt.ylabel("Sayı")
plt.savefig('Saç sütun sayısı')
plt.show()

sns.countplot(x="airborne", data=Animal)
plt.title('Hayvanlardaki doğum yapan sayısı')
plt.xlabel("Doğum")
plt.ylabel("Sayı")
plt.savefig('Doğum sütun sayısı.png')
plt.show()

sns.countplot(x="milk", data=Animal)
plt.title('Hayvanlardaki süt ile beslenme sayısı')
plt.xlabel("süt ile beslenme")
plt.ylabel("Sayı")
plt.savefig('Süt sütun sayısı.png')
plt.show()

sns.countplot(x="aquatic", data=Animal)
plt.title('Suda yaşayan hayvanların sayısı')
plt.xlabel("Suda yaşayan hayvanalar")
plt.ylabel("Sayı")
plt.savefig('Suda yaşayan sütun sayısı.png')
plt.show()

