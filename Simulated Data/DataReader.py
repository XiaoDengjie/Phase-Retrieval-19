# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:24:10 2019

@author: aholm
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load in CSV files
FIntensePD = pd.read_csv("FreqIntensity.csv")
TIntensePD = pd.read_csv("TimeIntensity.csv")
FreqPD = pd.read_csv("Frequency.csv")
TimePD = pd.read_csv("Time.csv")
PhisPD = pd.read_csv("Phis.csv")

print(PhisPD)
#Which dataset would you like to visualize
DataSetNumber = 0

#Load specific dataset line
f = np.array(FreqPD.iloc[0])
t = np.array(TimePD.iloc[0])
fIntense = np.array(FIntensePD.iloc[DataSetNumber])
tIntense = np.array(TIntensePD.iloc[DataSetNumber])
phis = np.array(PhisPD.iloc[DataSetNumber])

#Plot data
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10,8))

ax1.plot(f, fIntense, "k")
ax2.plot(t, tIntense, "k")

ax1.set_title('Intensity as a Function of Frequency')
ax1.set_xlabel('Frequency')
ax1.set_ylabel('Intensity')

ax2.set_title('Intensity as a Function of Time')
ax2.set_xlabel('Time')
ax2.set_ylabel('Intensity')

PhisPD.plot(table=True, ax=ax3)
plt.tight_layout()