#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:18:28 2019

@author: gzhou + aholm
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

class freq_Efield(object):

    def __init__(self, Aks,wks,phiks,tks,sigmaks,w):
        self.Aks = Aks
        self.wks = wks
        self.phiks = phiks
        self.tks = tks
        self.sigmaks = sigmaks
        self.modes_num = len(Aks)
        self.w = w
    def freq_single(self, Ak,wk,phik,tk,sigmak):
        #note: t should be an numpy array
        phase = -self.w*tk+phik 
        gaussian = -(sigmak**2*(self.w-wk)**2)/2
        Efield_single = Ak*sigmak*np.exp(1j*phase)*np.exp(gaussian)
        return Efield_single
    def freq_field(self):
        efield = np.zeros([self.modes_num,len(self.w)],dtype=complex)
        for i in range(self.modes_num):
            efield[i,:] = self.freq_single(self.Aks[i],self.wks[i],\
                  self.phiks[i],self.tks[i],self.sigmaks[i])
        return np.sum(efield,axis=0)


#Constants
e_charge = 1.602176565e-19
h_plank = 6.62607004e-34
c_speed = 299792458

#Inputs
ph_en = 8300 #in the uint of eV
w_cen =  ph_en * e_charge /h_plank *2* np.pi #frequency of radiation
n = 5 #Number of modes
#np.random.seed(1) #Set seed if want to create reproducable results

#Parameters
Aks = np.ones(n) #np.random.rayleigh(size=n) is true variation
wks = np.ones(n)*w_cen #np.random.normal(w_cen,10**-4*w_cen,n) true variation
phiks = np.random.random(n)*2*np.pi #random values between 0 and 2pi
tks = np.arange(1,n+1)*1e-15 #Will probably introduce variation later
sigmaks = np.ones(n)*0.3e-15 #fixed gaussian width
#Time domain
t=np.arange(0,(n+1)*1e-15,12e-18)
#Frequency Domain
x = 1000 #Number of points to test in frequeny domain
wrange = np.array([1-1e-3,1+1e-3])*w_cen #Domain centered around central frequency
w=np.arange(wrange[0],wrange[1],(wrange[1]-wrange[0])/x)

'''
Way to generate electric field in time domain
efieldtmpt = time_Efield(Aks,wks,phiks,tks,sigmaks,t)
tfield = efieldtmpt.time_field()
plt.plot(t,abs(tfield)**2)

Way to generate electric field in frequency domain
efieldtmpf = freq_Efield(Aks,wks,phiks,tks,sigmaks,w)
ffield = efieldtmpf.freq_field()
plt.plot(w,abs(ffield)**2)

Important Note is that we are dealing with Intensity which is the E
field modulus squared
'''

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
    efieldtmpf = freq_Efield(Aks,wks,phiks,tks,sigmaks,w)
    ffield = efieldtmpf.freq_field()
    Iw = np.abs(ffield)**2
    phid = pd.DataFrame(phiks).T
    Iwd = pd.DataFrame(Iw).T
    Itd = pd.DataFrame(It).T
    PhiData = PhiData.append(phid)
    IntensityFreqData = IntensityFreqData.append(Iwd)
    IntensityTimeData = IntensityTimeData.append(Itd)
    print(i)

#Also will make a dataset to encode domains
FrequencyData = pd.DataFrame(w).T
TimeData = pd.DataFrame(t).T

#Rename columns annd rows for better readability
for i in range(n):
    PhiData = PhiData.rename(columns={i:'Phi' + str(i+1)})
for i in range(x):
    IntensityTimeData = IntensityTimeData.rename(columns={i:'I' + str(i+1)})
    IntensityFreqData = IntensityFreqData.rename(columns={i:'I' + str(i+1)})
    TimeData = TimeData.rename(columns={i:'t' + str(i+1)})
    FrequencyData = FrequencyData.rename(columns={i:'w' + str(i+1)})

#Print out DataFrames to csv files
IntensityFreqData.to_csv('FreqIntensity.csv', index=False)
IntensityTimeData.to_csv('TimeIntensity.csv', index=False)
PhiData.to_csv('Phis.csv', index=False)
FrequencyData.to_csv('Frequency.csv', index=False)
TimeData.to_csv('Time.csv', index=False)

