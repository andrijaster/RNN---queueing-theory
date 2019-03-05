# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:47:46 2018

@author: Andrija Master
"""

import numpy as np
import pandas as pd
import forecastio
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#api_key= '969ee369b45b33290490105dfcd6e627'
#api_key= '9405dd161a4351820b2134b9140c5cd5'
api_key= 'a81f7318d943521ba62906a79eed4ab7'
#api_key = '765b5ab60ef98411ce5268c46c3f4d54'
    


#lat = np.array([45.051666, 44.866667]) # Sid - Simanovci
#long = np.array([19.126166, 20.083333]) # Sid - Simanovci
lat = np.array([43.3, 44.816667]) # Nis - Beograd
long = np.array([21.9, 20.466667]) # Nis - Beograd
pocetak = '2017-08-01 00:00:00'
kraj = '2017-12-01 00:00:00'   
poceta = pd.to_datetime(pocetak)
kraj = pd.to_datetime(kraj) 
index1 = pd.date_range(start = pocetak, end = kraj, freq = '1D') 
index2 = pd.date_range(start = pocetak, end = kraj, freq = '1H') 
index = pd.date_range(start = pocetak, end = kraj, freq = '5min') 
dataset = pd.DataFrame(np.zeros([len(index),11]), index = index)
pomocna0 = []
pomocna1 = []
for i in index1:
    for j in range(lat.shape[0]):
        sirina = lat[j]
        duzina = long[j]
        forecast = forecastio.load_forecast(api_key, sirina, duzina,time=i)
        byHour = forecast.hourly()
        for hourlyData in byHour.data:
            if j==0:
                pomocna0.append([hourlyData.temperature,hourlyData.dewPoint,hourlyData.humidity,
                                 hourlyData.icon, hourlyData.visibility])
#                pomocna0.append([hourlyData.temperature,hourlyData.dewPoint,hourlyData.humidity, 
#                      hourlyData.cloudCover,hourlyData.windSpeed])
            else:
                pomocna1.append([hourlyData.temperature,hourlyData.dewPoint,hourlyData.humidity,
                                 hourlyData.icon,hourlyData.visibility,hourlyData.time])
#                pomocna1.append([hourlyData.temperature,hourlyData.dewPoint,hourlyData.humidity, 
#                      hourlyData.cloudCover,hourlyData.windSpeed])
pomocna0 = np.asarray(pomocna0)
pomocna1 = np.asarray(pomocna1)
vreme = np.concatenate((pomocna0,pomocna1), axis = 1)
dataset2 = pd.DataFrame(vreme[:-24,:], index = index2)

for i in range(dataset2.shape[0]):
    if i==dataset2.shape[0]-1:
        dataset[dataset.index == dataset2.index[i]] = [dataset2.iloc[i,:].values]
    else:
        vektor = dataset2.iloc[i,:].values
        uslov = (dataset.index >= dataset2.index[i]) & (dataset.index < dataset2.index[i+1])
        dataset[uslov] = np.tile(vektor,[12,1])

dataset['Vreme'] = dataset.index
dataset['Vreme'] = dataset['Vreme'].apply(lambda x: x.hour*60+x.minute)        
kolone = {0:'temperature',1:'dew-point',2:'humidity',3:'icon',4:'visibility',
          5:'temperature-1',6:'dew-point-1',7:'humidity-1',8:'icon-1',9:'visibility-1',10:'time'}
dataset.rename(columns=kolone, inplace = True)
dataset.drop(['time'],axis=1,inplace=True)

enc = OneHotEncoder(handle_unknown='ignore',sparse = False)
label = LabelEncoder()
x = label.fit_transform(dataset.icon)
x = x.reshape(len(x), 1)
x = enc.fit_transform(x)
x1 = label.fit_transform(dataset.loc[:,'icon-1'])
x1 = x1.reshape(len(x1), 1)
x1 = enc.fit_transform(x1)
novo = np.concatenate((x,x1), axis=1)
novidf = pd.DataFrame(novo, index = index)
dataset.drop(['icon'],axis=1,inplace=True)
dataset.drop(['icon-1'],axis=1,inplace=True)
dataset = pd.concat([dataset,novidf],axis=1)
dataset.to_csv('dataset_NiSBG1.csv')
        
