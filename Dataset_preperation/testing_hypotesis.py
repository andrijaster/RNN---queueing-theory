# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:25:06 2019

@author: Andri
"""

import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
import warnings
import math
warnings.filterwarnings('ignore')


""" Slaganje podataka """
mu_vreme = pd.read_csv("mu_vreme.csv", index_col = 0)
br_voz_min = pd.read_csv('vreme_izmedju_voz.csv', index_col = 0)
minimum2 = 1/np.min(br_voz_min['br_voz_min'])


kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(mu_vreme.values)
x = np.linspace(mu_vreme.values.min(), 10, num = 1000)
pdf = np.exp(kde.score_samples(x.reshape(-1,1)))


fig2 = plt.figure()
ax2 = plt.subplot(111)
ax2.plot(x, pdf,'black', lw=2, alpha=1, label='Kernel Density Estimator')
#ax2.hist(mu_vreme, bins = 10, label = 'emp')
ax2.legend()
ax2.grid()

mu_vreme = mu_vreme[0.3<mu_vreme.iloc[:,0]]
mu_vreme_pomoc = mu_vreme[mu_vreme.iloc[:,0]<minimum2]
variance = np.var(mu_vreme_pomoc.values)
minimum = minimum2+3*np.sqrt(variance)
mu_vreme = mu_vreme[mu_vreme.iloc[:,0]<minimum]


velicina = mu_vreme.size
size_example_redovi = 150
size_example_sample = 150000


""" Fitovanje """
bin_no = int(5*math.log10(size_example_redovi))
data_org = mu_vreme.iloc[:,0].values
redovi = np.random.randint(low = 0, high = velicina, size = size_example_redovi)
data_org = data_org[redovi]

loc_exp, scale_exp = sp.expon.fit(mu_vreme)
dist = sp.expon(loc = loc_exp, scale = scale_exp)
data1 = dist.rvs(size_example_sample)
data2 = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 500)



""" Testiranje """
# Kolmogorov Smirnov
D, p_value = sp.ks_2samp(data1,data_org)
D1, p_value1 = sp.kstest(data_org, dist.cdf, N = 100)



""" Testiranje Chi2"""
#hist, edges = np.histogram(data_org, bins=bin_no, density=True)
#hist_exp, edges_exp = np.histogram(data1, bins=bin_no, range=(edges[0],edges[-1]) , density=True)

hist1, edges = np.histogram(data_org, bins=bin_no, density=False)
hist2 = hist1.copy()
hist1 = hist1/np.sum(hist1)
hist_exp1, edges_exp = np.histogram(data1, bins=bin_no, range=(edges[0],edges[-1]) , density=False)
hist_exp1 = hist_exp1/np.sum(hist_exp1)

# DRUGI NACIN
#hist_exp, edges_exp = np.histogram(data1, bins=bin_no, density=True)
#hist, edges = np.histogram(data_org, bins=bin_no, range=(edges_exp[0],edges_exp[-1]), density=True)
#
#hist_exp1, edges_exp = np.histogram(data1, bins=bin_no, density=True)
#hist1, edges = np.histogram(data_org, bins=bin_no, range=(edges_exp[0],edges_exp[-1]), density=True)

edges[-1] = 1e8
f_exp = np.diff(sp.expon.cdf(edges, loc = loc_exp, scale = scale_exp))
f_exp1 = f_exp*size_example_redovi

KL_pq = sp.entropy(f_exp, hist1)
KL_qp = sp.entropy(hist1, f_exp)

chisq, p_value3 = sp.chisquare(hist2, f_exp1)

DKL = 0.5*KL_pq + 0.5*KL_qp


""" Ploting """
fig = plt.figure()
ax = plt.subplot(111)
ax.set_xlim(0.3, 1.3)


ax.hist(data_org, bins = edges, label = 'emp', alpha = 0.5, histtype ='bar', ec = 'black', density = True)
#ax.hist(data1, bins = edges, density = True, lw=5, alpha=0.5, label='expon pdf')

ax.plot(data2, dist.pdf(data2),'r-', lw=5, alpha=0.5, label='expon pdf')
ax.legend(loc='best')
ax.grid()





print('KS2 - D, p_value = ', D, p_value)
print('KS - D, p_value = ', D1, p_value1)
print('chisq - D, p_value = ', chisq, p_value3)
print('KL_pq = ', KL_pq)
print('1-np.exp(-KL_pq) = ', 1-np.exp(-KL_pq))
print('KL_pq = ', KL_qp)
print('1-np.exp(-KL_qp) = ', 1-np.exp(-KL_qp))
print('DKL = ', DKL)
print('1-np.exp(-DKL) = ', 1-np.exp(-DKL))




