# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:25:08 2019

@author: aholm
"""

import numpy as np
from matplotlib import pyplot as plt
from Generator import tfield, ffield_fft
from GSAlg import GSFreq, LossFunctionTime, LossFunctionFreq

TimeElectricAmp = np.abs(tfield)
FreqElectricAmp = np.abs(ffield_fft)

#Generate lots of data points to observe best initial condition
z = 1000
TimeDiff = np.zeros(z)
FreqDiff = np.zeros(z)

#Calculate loss function in both time and freq domain for each generated seed
for i in range(z):
    Et, Ew = GSFreq(20,TimeElectricAmp,FreqElectricAmp,i)
    TimeDiff[i] = LossFunctionTime(tfield,Et)
    FreqDiff[i] = LossFunctionFreq(ffield_fft,Ew)

#Plot different error in time and freq space. Because fourier pair, same shape
#Slight difference because manually chose region of interest
fig1, axs1 = plt.subplots(2, 1)
axs1[0].plot(TimeDiff)
axs1[1].plot(FreqDiff)

#Check how many loss functions are below some threshold
Threshold = 0.3
NumberBelow = 0
for i in range(z):
    if TimeDiff[i] <= Threshold:
        NumberBelow += 1
print(NumberBelow)

#Results
Peaks = [1,2,3,4,5,6,7,8,9,10]
Below3 = [183,202,198,184,41,35,28,30,17,24]
Below2 = [102,125,116,109,11,10,8,15,7,2]
Below1 = [3,21,40,23,0,0,0,0,0,0]

fig2, axs2 = plt.subplots(1, 3)
axs2[0].set_ylabel('Number')
axs2[0].set_xlabel('Number of Peaks')
axs2[0].set_title('Number of Tests out of 1000: Below 0.3')
axs2[0].plot(Peaks,Below3)
axs2[1].set_ylabel('Number')
axs2[1].set_xlabel('Number of Peaks')
axs2[1].set_title('Number of Tests out of 1000: Below 0.2')
axs2[1].plot(Peaks,Below2)
axs2[2].set_ylabel('Number')
axs2[2].set_xlabel('Number of Peaks')
axs2[2].set_title('Number of Tests out of 1000: Below 0.1')
axs2[2].plot(Peaks,Below1)





