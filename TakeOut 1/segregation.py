# -*- coding: utf-8 -*-
"""
Author - Jigyasa Singh
Take Out 1 - Facial Expression/Emotion Detection
This program focuses on cleaning the Fer2013 CSV file by removing rendundant data and segregating the emotions into 7 categories :
'Angry', 'Disgust','Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'

"""
import matplotlib
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from matplotlib import pyplot as plt
import seaborn as sns

x = pd.read_csv('fer2013.csv')
print('Original Dataset:', x.values.shape)

# Cleaning - removing duplicate pixel data
# x.drop_duplicates(subset ="pixels", keep = False, inplace = True)
x.drop_duplicates(inplace=True)
print('Modified Dataset:', x.values.shape)
df_0 = x[x.emotion == 0]
df_1 = x[x.emotion == 1]
df_2 = x[x.emotion == 2]
df_3 = x[x.emotion == 3]
df_4 = x[x.emotion == 4]
df_5 = x[x.emotion == 5]
df_6 = x[x.emotion == 6]

emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
e_angry = x[x.emotion == 0].count()
print('Angry emotions in dataset: ', e_angry[0])
e_disgust = x[x.emotion == 1].count()
print('Disgust emotions in dataset: ', e_disgust[0])
e_fear = x[x.emotion == 2].count()
print('Fear emotions in dataset: ', e_fear[0])
e_happy = x[x.emotion == 3].count()
print('Happy emotions in dataset: ', e_happy[0])
e_sad = x[x.emotion == 4].count()
print('Sad emotions in dataset: ', e_sad[0])
e_surprise = x[x.emotion == 5].count()
print('Surprise emotions in dataset: ', e_surprise[0])
e_neutral = x[x.emotion == 6].count()
print('Neutral emotions in dataset: ', e_neutral[0])
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))
sns.countplot(data=x, x='emotion', ax=ax1).set_title('Emotions in Dataset')
ax1.set_xticklabels(emotions.values())
data = x.values

# Y being the Dependent variable: Emotion data {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
y = data[:, 0]  
pixels = data[:, 1]

# X being the Independent variable: Face data
X = np.zeros((pixels.shape[0], 48 * 48))  
plt.show()

#plt.savefig( 'Segregation chart.png' )
for ix in range(X.shape[0]):
    p = pixels[ix].split(' ')
    for iy in range(X.shape[1]):
        X[ix, iy] = int(p[iy])
