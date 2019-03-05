# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:39:26 2019

@author: Andrija Master
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')





pocetak = '2017-08-01 00:00:00'
kraj = '2017-12-01 00:00:00'  
pocetak = pd.to_datetime(pocetak)
kraj = pd.to_datetime(kraj) 


kolone_lev1 = ['broj_ljudi','brzina_sred','brzina_median','brzina_10','brzina_25','brzina_75','brzina_90','brzina_100',
               'lambda']

kolone = [kolone_lev1]
kolone = pd.MultiIndex.from_product(kolone)
oznake = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
#df1 = pd.read_csv('012017_bgd-nis_1.csv', sep=';', header = None)
#df2 = pd.read_csv('012017_bgd-nis_2.csv', sep=';', header = None)
#df3 = pd.read_csv('022017_bgd-nis_1.csv', sep=';', header = None)
#df4 = pd.read_csv('022017_bgd-nis_2.csv', sep=';', header = None)
#df5 = pd.read_csv('032017_bgd-nis_1.csv', sep=';', header = None)
#df6 = pd.read_csv('032017_bgd-nis_2.csv', sep=';', header = None)
#df7 = pd.read_csv('042017_bgd-nis_1.csv', sep=';', header = None)
#df8 = pd.read_csv('042017_bgd-nis_2.csv', sep=';', header = None)
#df9 = pd.read_csv('052017_bgd-nis_1.csv', sep=';', header = None)
#df10 = pd.read_csv('052017_bgd-nis_2.csv', sep=';', header = None)
#df11 = pd.read_csv('062017_bgd-nis_1.csv', sep=';', header = None)
#df12 = pd.read_csv('062017_bgd-nis_2.csv', sep=';', header = None)
#df13 = pd.read_csv('072017_bgd-nis_1.csv', sep=';', header = None)
#df14 = pd.read_csv('072017_bgd-nis_2.csv', sep=';', header = None)
#df15 = pd.read_csv('072017_bgd-nis_3.csv', sep=';', header = None)
df16 = pd.read_csv('082017_bgd-nis_1.csv', sep=';', header = None)
df17 = pd.read_csv('082017_bgd-nis_2.csv', sep=';', header = None)
df18 = pd.read_csv('082017_bgd-nis_3.csv', sep=';', header = None)
df19 = pd.read_csv('092017_bgd-nis_1.csv', sep=';', header = None)
df20 = pd.read_csv('092017_bgd-nis_2.csv', sep=';', header = None)
df21 = pd.read_csv('102017_bgd-nis_1.csv', sep=';', header = None)
df22 = pd.read_csv('102017_bgd-nis_2.csv', sep=';', header = None)
df23 = pd.read_csv('112017_bgd-nis_1.csv', sep=';', header = None)
df24 = pd.read_csv('112017_bgd-nis_2.csv', sep=';', header = None)
df25 = pd.read_csv('122017_bgd-nis_1.csv', sep=';', header = None)
#df26 = pd.read_csv('122017_bgd-nis_2.csv', sep=';', header = None)
frames = [df16, df17, df18, df19, df20, df21, df22, df23, df24, df25]
result = pd.concat(frames)
svekolone_iz = 1
svekolone_ul = np.array([1,6,7,9,51,11,12,13,14,16,18,20,21,23,25,26,27,28,29])
mapiranje = {1:0,6:1,7:2,9:3,51:4,11:5,12:6,13:7,14:8,16:9,18:10,20:11,21:12,23:13,25:14,26:15,27:16,28:17,29:18}
result.drop(labels = [0,3,5],axis=1, inplace = True)
kolone = {1:'Ulaz-stan',2:'Ulaz-traka',4:'date1',6:'Izlaz-stan',7:'Izlaz-traka',8:'date2',9:'Kategorija',10:'tag'}
result.rename(columns=kolone, inplace = True)
result = result[result['Ulaz-stan'].isin(svekolone_ul)]
result = result[result['Izlaz-stan']==1]
result = result[result['tag']==0]
result['Ulaz-stan'] = result['Ulaz-stan'].map(mapiranje)
result['Izlaz-stan'] = result['Izlaz-stan'].map(mapiranje)
result.dropna(axis=0,inplace = True)
result['Ulaz-stan'] = result['Ulaz-stan'].astype(int)
result['Izlaz-stan'] = result['Izlaz-stan'].astype(int)
result = result[result.loc[:,'date1'] != '.  .       :  :']
result = result[result.loc[:,'date2'] != '.  .       :  :']
result.sort_values('date2',axis=0,inplace = True)
result = result.iloc[:,:]
result['date2'] = pd.to_datetime(result['date2'],dayfirst = True)
result['date1'] = pd.to_datetime(result['date1'],dayfirst = True)
result.set_index('date2',inplace = True,drop = False)
result = result[result.loc[:,'Ulaz-stan']>result.loc[:,'Izlaz-stan']]
di = {2:1,3:1}
result.replace({'tag':di},inplace = True)
result['timeDELTA'] = result['date2']-result['date1']
result['timeDELTA'] = result['timeDELTA'].dt.total_seconds()/3600
result = result[result['timeDELTA']!=0]
result = result[result['timeDELTA']<10]
result.sort_values(by='Izlaz-stan',inplace = True)
Kilometraza = pd.read_csv('kilometraza_nis.csv',header=None,sep=';')
result.reset_index(inplace = True, drop = True)
grupe = result.groupby(['Ulaz-stan','Izlaz-stan'])
result['kilometraza']=0
for key,value in grupe:
    value['kilometraza'] = Kilometraza.iloc[key[1],key[0]]
    result['kilometraza'][result.index.isin(value.index)] = value['kilometraza']
result['brzina'] = result['kilometraza']/result['timeDELTA']
result = result[result['brzina']>=0]
result.sort_values('date2',axis=0,inplace = True)
result.set_index('date2',inplace = True, drop = True)
uslov = (result.index >= pocetak) & (result.index <= kraj)
result = result[uslov]
result.to_csv('result.csv', header = True, index = True)