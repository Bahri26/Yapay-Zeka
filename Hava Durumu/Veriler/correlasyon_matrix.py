# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:21:31 2019

@author: Casper
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


veri = pd.read_csv('Weather_predict.csv')
del veri['id']

veri2 = veri.apply(LabelEncoder().fit_transform)

x =veri2.iloc[:,0:9]
y =veri2.iloc[:,9:]
X =x.values
Y =y.values

print("----Değişkenlerin sınıf  değerine etkisi----------")

corr1 = veri.corr()
fig, ax = plt.subplots()
im = ax.imshow(corr1.values)

ax.set_xticks(np.arange(len(corr1.columns)))
ax.set_yticks(np.arange(len(corr1.columns)))
ax.set_xticklabels(corr1.columns)
ax.set_yticklabels(corr1.columns)

plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
rotation_mode="anchor")
sns.heatmap(veri.corr(), annot=False)
plt.savefig('correlaasyon_matrix.png')
plt.show()