# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 20:07:43 2018

@author: Andrija Master
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy as sp
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def atributi(brojminuta,result,dataset1):
    
    for i in range(dataset1.shape[0]):
        uslov = (result.index >= dataset1.index[i] - brojminuta ) & (result.index <= dataset1.index[i]) 
        grupe = result[uslov]
        broj_traka = grupe['Izlaz-traka'].nunique()
        broj_ljudi = grupe.shape[0]
        brzina_sred = grupe['brzina'].mean()
        brzina_median = grupe['brzina'].median()
        brzina_10 = grupe['brzina'].quantile(0.1)
        brzina_25 = grupe['brzina'].quantile(0.25)
        brzina_75 = grupe['brzina'].quantile(0.75)
        brzina_90 = grupe['brzina'].quantile(0.9)
        brzina_100 = grupe['brzina'].quantile(1)
        if broj_traka == 0:
            dataset1.ix[i,'broj_ljudi'] = 0
        else:
            dataset1.ix[i,'broj_ljudi'] = broj_ljudi
        dataset1.ix[i,'brzina_sred'] = brzina_sred
        dataset1.ix[i,'brzina_median'] = brzina_median
        dataset1.ix[i,'brzina_10'] = brzina_10
        dataset1.ix[i,'brzina_25'] = brzina_25
        dataset1.ix[i,'brzina_75'] = brzina_75
        dataset1.ix[i,'brzina_90'] = brzina_90
        dataset1.ix[i,'brzina_100'] = brzina_100
        dataset1.ix[i,'lambda'] = 1 / grupe['vreme_izmedju'].mean()
        dataset1.ix[i,'broj_traka'] = broj_traka
    
    return dataset1

def atributi2(brojminuta2,brojminutax2,result,br_voz_min):
    for i in range(br_voz_min.shape[0]):
        uslov = (result.index >= br_voz_min.index[i] - brojminuta2 ) & (result.index <= br_voz_min.index[i]) 
        grupe = result[uslov]
        broj_traka = grupe['Izlaz-traka'].nunique()
        if broj_traka == 0:  
            br_voz_min.ix[i,'br_voz_min'] = 0
        else:      
            broj_ljudi = grupe.shape[0]/broj_traka
            br_voz_min.ix[i,'br_voz_min'] = broj_ljudi/brojminutax2
    return br_voz_min

def outputs(lambde,lambda_koraci,atributi_lambde,no_steps):
    for i in range(lambda_koraci.shape[0]):
#        print(i)
        if i<=lambda_koraci.shape[0]-no_steps:
            lambda_koraci.values[i,:] = lambde[i:i+no_steps]
        if i>=no_steps:
            atributi_lambde.values[i,:] = lambde[i-no_steps:i]
    return lambda_koraci, atributi_lambde
            
    
brojminuta = 180
brojminuta2 = 0.75  
pocetak = '2017-08-01 00:00:00'
kraj = '2017-12-01 00:00:00'  
poceta = pd.to_datetime(pocetak)
kraj = pd.to_datetime(kraj) 
index2 = pd.date_range(start = pocetak, end = kraj, freq = '0.75min') 
brojminutax2 = brojminuta2
brojkoraka = int(brojminuta/5)


dataset = pd.read_csv("dataset_NiSBG1.csv", header = 0, index_col = 0)
dataset.index = pd.to_datetime(dataset.index)
vreme_izmedju_voz = pd.DataFrame(np.zeros([len(index2),1]), index = index2)

kolone_lev1 = ['broj_ljudi','brzina_sred','brzina_median','brzina_10','brzina_25','brzina_75','brzina_90','brzina_100',
               'lambda','broj_traka']


result = pd.read_csv("result.csv", header = 0)
duzina = len(result.index) - 2
result['date2'] = pd.to_datetime(result['date2'],dayfirst = True)
razlika = (result.ix[1:,'date2'].values-result.ix[:duzina,'date2']).dt.total_seconds()/60
result['vreme_izmedju'] = razlika
result.set_index('date2',inplace = True, drop = False)

brojminuta = pd.Timedelta(minutes = brojminuta)
brojminuta2 = pd.Timedelta(minutes = brojminuta2)
dataset = atributi(brojminuta,result,dataset)
broj_slobodnih_traka = dataset['broj_traka']
#dataset.drop('broj_traka', inplace = True)


vreme_izmedju_voz = atributi2(brojminuta2,brojminutax2,result,vreme_izmedju_voz)
lambde = dataset['lambda']
lambde = np.squeeze(lambde.transpose())
result_stan = result.loc[result.loc[:,'Izlaz-traka']==16]

""" Evaluacija mu vrednosti """

vreme_izmedju_voz = atributi2(brojminuta2, brojminutax2, result_stan, vreme_izmedju_voz)
duzina2 = len(result_stan.index)-1
mu_vreme = (result_stan.ix[1:,'date2'].values-result_stan.ix[:duzina2,'date2']).dt.total_seconds()/60
vreme_izmedju_voz = vreme_izmedju_voz[vreme_izmedju_voz['br_voz_min']>0]

dataset.to_csv('dataset.csv')
vreme_izmedju_voz.to_csv('vreme_izmedju_voz.csv')
mu_vreme.to_csv('mu_vreme.csv')
    





