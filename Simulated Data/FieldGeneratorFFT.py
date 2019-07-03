# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 20:41:47 2019

@author: aholm
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime

class time_Efield(object):
    #Generate the electric field in the time domain.
    def __init__(self, Aks,wks,phiks,tks,sigmaks,t):
        #Amplitude of each gaussian
        self.Aks = Aks
        #Frequencies at which each wave is generated at
        self.wks = wks
        #Phase term. This is what we will be attempting to retrieve through ML
        self.phiks = phiks
        #Time in which pulse occurred.
        self.tks = tks
        #Width of gaussian
        self.sigmaks = sigmaks
        #Number of pulses
        self.modes_num = len(Aks)
        #Time domain
        self.t = t
    def time_single(self, Ak,wk,phik,tk,sigmak):
        #note: t should be an numpy array
        phase = wk*self.t+phik
        gaussian = -((self.t-tk)**2)/2/sigmak**2
        Efield_single = Ak*np.exp(1j*phase)*np.exp(gaussian)
        return Efield_single
    def time_field(self):
        efield = np.zeros([self.modes_num,len(self.t)],dtype=complex)
        for i in range(self.modes_num):
            efield[i,:] = self.time_single(self.Aks[i],self.wks[i],\
                  self.phiks[i],self.tks[i],self.sigmaks[i])
        return np.sum(efield,axis=0)

#Constants
e_charge = 1.602176565e-19
h_plank = 6.62607004e-34
c_speed = 299792458
x = 1000 #Number of points to test in frequeny and time domain

#Inputs
ph_en = 8300 #in the uint of eV
w_cen =  ph_en * e_charge /h_plank *2* np.pi #frequency of radiation
n = 3 #Number of modes
#np.random.seed(1) #Set seed if want to create reproducable results

#Parameters
Aks = np.ones(n) #np.random.rayleigh(size=n) is true variation
wks = np.ones(n)*w_cen #np.random.normal(w_cen,10**-4*w_cen,n) true variation
phiks = np.random.random(n)*2*np.pi #random values between 0 and 2pi
tks = np.arange(1,n+1)*1e-15 #Will probably introduce variation later
sigmaks = np.ones(n)*0.3e-15 #fixed gaussian width

#Time domain
t=np.linspace(-5e-15,(n+1)*1e-15+5e-15,x)
freq = np.fft.fftfreq(len(tfield))

#Create empty Pandas DataFrames to append to with data
PhiData = pd.DataFrame()
IntensityFreqData = pd.DataFrame()
IntensityTimeData = pd.DataFrame()

#Function to generate data
z = 10 #Number of datasets generated
for i in range(z):
    phiks = np.random.random(n)*2*np.pi
    efieldtmpt = time_Efield(Aks,wks,phiks,tks,sigmaks,t)
    tfield = efieldtmpt.time_field()
    It = np.abs(tfield)**2
    Iw =  np.abs(np.fft.fft(tfield))**2
    phid = pd.DataFrame(phiks).T
    Iwd = pd.DataFrame(Iw).T
    Itd = pd.DataFrame(It).T
    PhiData = PhiData.append(phid)
    IntensityFreqData = IntensityFreqData.append(Iwd)
    IntensityTimeData = IntensityTimeData.append(Itd)
    print(i)

#Also will make a dataset to encode domains
TimeData = pd.DataFrame(t).T
FrequencyData = pd.DataFrame(freq).T
#Rename columns annd rows for better readability
for i in range(n):
    PhiData = PhiData.rename(columns={i:'Phi' + str(i+1)})
for i in range(x):
    IntensityTimeData = IntensityTimeData.rename(columns={i:'I' + str(i+1)})
    IntensityFreqData = IntensityFreqData.rename(columns={i:'I' + str(i+1)})
    TimeData = TimeData.rename(columns={i:'t' + str(i+1)})
    FrequencyData = FrequencyData.rename(columns={i:'w' + str(i+1)})

#Retrieve date to automatically create directory
now = datetime.datetime.now()
month = now.strftime("%B")
day = now.strftime("%d")

directory = month + "-" + day + " FFT"
current_directory = os.getcwd()
print(current_directory)
final_directory = os.path.join(current_directory + "\Dataset", directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

#Print out DataFrames to csv files
IntensityFreqData.to_csv(final_directory + '\FreqIntensity.csv', index=False)
IntensityTimeData.to_csv(final_directory + '\TimeIntensity.csv', index=False)
PhiData.to_csv(final_directory + '\Phis.csv', index=False)
FrequencyData.to_csv(final_directory + '\Frequency.csv', index=False)
TimeData.to_csv(final_directory + '\Time.csv', index=False)