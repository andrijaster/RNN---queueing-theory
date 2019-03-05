# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:25:06 2019

@author: Andri
"""

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

mu_vreme = pd.read_csv("mu_vreme.csv", index_col = 0)
br_voz_min = pd.read_csv('vreme_izmedju_voz.csv', index_col = 0)

minimum = 1/np.min(br_voz_min['br_voz_min'])

mu_vreme = mu_vreme[mu_vreme.iloc[:,0]<minimum]
mu_vreme = mu_vreme[0.3<mu_vreme.iloc[:,0]]

plt.hist(mu_vreme.iloc[:,0], bins = 8)