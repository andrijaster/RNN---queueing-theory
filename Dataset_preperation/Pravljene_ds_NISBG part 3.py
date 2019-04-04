# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:25:39 2019

@author: Andri
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
import scipy as sp
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


def out_atr(lambde,lambda_koraci,atributi_lambde,no_steps):
    for i in range(lambda_koraci.shape[0]):
        if i<=lambda_koraci.shape[0]-no_steps:
            lambda_koraci.values[i,:] = lambde[i:i+no_steps]
        if i>=no_steps:
            atributi_lambde.values[i,:] = lambde[i-no_steps:i]
    return lambda_koraci, atributi_lambde


""" Ucitavanjae dataset-a """
dataset = pd.read_csv("dataset.csv", header = 0, index_col = 0)
dataset.index = pd.to_datetime(dataset.index,dayfirst=True)
dataset = dataset.iloc[1:,:]
series = dataset.loc[:,'lambda']
dataset.index[0]

""" Plotovoanje i diferenciranje """

figure = plt.figure(figsize=(13, 6))
ax1 = plt.subplot(211)
ax1.plot(series, 'black')
ax1.set_xlim(dataset.index[0],dataset.index[-1])
ax1.set_ylabel('$\mathbf{\lambda}$ [1/min]')
ax1.grid()

ax2 = plt.subplot(212)
diff = series.diff()
dataset['lambda_diff'] = diff
diff = diff.iloc[1:]
dataset1 = dataset.iloc[1:,:]
ax2.plot(diff, 'black')
ax2.set_xlim(dataset.index[0],dataset.index[-1])
ax2.set_ylabel('$\mathbf{\lambda_i} - \mathbf{\lambda_{i-1}}$ [1/min]')
ax2.grid()

""" Deljenje i pravljenje novih atributa """

brojminutaunapred = 200*5 # zadati vreme predvidjanja
no_steps = int(brojminutaunapred/5) # broj koraka unpared za output
no_steps_atribute = no_steps # broj koraka unpared za atribute


atributi_lambde = pd.DataFrame(np.zeros([dataset1.shape[0],no_steps_atribute]),index = dataset1.index)
output = pd.DataFrame(np.zeros([dataset1.shape[0],no_steps]),index = dataset1.index)
output, atributi_lambde = out_atr(diff,output,atributi_lambde,no_steps)
output = output.iloc[:-no_steps,:]
output = output.iloc[no_steps:,:]
atributi_lambde = atributi_lambde.iloc[no_steps:,:]
atributi_lambde = atributi_lambde.iloc[:-no_steps]
dataset = dataset.iloc[:-no_steps,:]
dataset = dataset.iloc[no_steps:,:]

atributes = dataset1.iloc[:,:31].join(atributi_lambde, how = 'inner')

""" Stvarni_izlaz """
output_ext = output.copy()
output_ext.insert(loc = 0, column = 'lambda' , value = dataset.ix[:,'lambda'])
output_ext['lambda'] = output_ext['lambda'].shift(+1)
output_ext.ix[0,'lambda'] = dataset.ix[0,'lambda']


name_atr = os.path.join("C:\\Users\\Andri\\Documents\\GitHub\\RNN---queueing-theory---maximum-principle\\LSTM_problem","Atribute.csv")
name_out = os.path.join("C:\\Users\\Andri\\Documents\\GitHub\\RNN---queueing-theory---maximum-principle\\LSTM_problem","Output.csv")
name_out_ext = os.path.join("C:\\Users\\Andri\\Documents\\GitHub\\RNN---queueing-theory---maximum-principle\\LSTM_problem","Output_ext.csv")

atributes.to_csv(name_atr)
output.to_csv(name_out)
output_ext.to_csv(name_out_ext)



