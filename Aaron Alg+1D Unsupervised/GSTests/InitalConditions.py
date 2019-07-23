# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:25:08 2019

@author: aholm
"""

import numpy as np
from matplotlib import pyplot as plt
from Generator import tfield, ffield_fft
from GSAlg import GSFreq

TimeElectricAmp = np.abs(tfield)
FreqElectricAmp = np.abs(ffield_fft)

#Generate lots of data points to observe best initial condition
z = 100
All_TimeDiff = np.zeros(z)
All_FreqDiff = np.zeros(z)

#Seed 90 and 91 are good. Seed 92 is bad.
Et,Ew = GSFreq(20,TimeElectricAmp,FreqElectricAmp,90)

def Normalize(actual,guess):
    numer = np.abs(guess - actual)
    denom = np.sqrt(2)*np.sqrt(np.abs(guess)**2 + np.abs(actual)**2)
    Normal = np.zeros(len(numer))
    for i in range(800,1200):
        Normal[i] = numer[i]/denom[i]
    return Normal,numer,denom

Normal, numer, denom = Normalize(tfield,Et)

plt.figure()
plt.plot(numer)
plt.figure()
plt.plot(denom)
plt.figure()
plt.plot(Normal)

print(np.linalg.norm(Normal))


'''
for i in range(z):
    Et, Ew = GSFreq(50,TimeElectricAmp,FreqElectricAmp,i)
    TimeDifference = np.linalg.norm(Normalize(tfield,Et))
    FreqDifference = np.linalg.norm(Normalize(ffield,Ew))
    All_TimeDiff[i] = TimeDifference
    All_FreqDiff[i] = FreqDifference

#Plot different error in time and freq space. Because fourier pair, same shape
fig3, axs3 = plt.subplots(2, 1)
axs3[0].plot(All_TimeDiff)
axs3[1].plot(All_FreqDiff)
'''