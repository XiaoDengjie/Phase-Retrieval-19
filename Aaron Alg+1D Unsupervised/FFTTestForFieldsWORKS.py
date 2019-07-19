# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:12:33 2019

@author: aholm
"""

import numpy as np
from matplotlib import pyplot as plt

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
        phase = -self.w*tk+phik+wk*tk
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
x = 2001 #Number of points to test in frequeny and time domain

#Inputs
ph_en = 8300 #in the uint of eV
w_cen =  ph_en * e_charge /h_plank *2* np.pi #frequency of radiation
n = 2 #Number of modes
np.random.seed(1) #Set seed if want to create reproducable results

#Parameters
Aks = np.ones(n) #np.random.rayleigh(size=n) is true variation
wks = np.ones(n)*w_cen #np.random.normal(w_cen,10**-4*w_cen,n) true variation
phiks = np.random.random(n)*2*np.pi #random values between 0 and 2pi
tks = np.arange(0,n)*1e-15 - .5e-15 #Will probably introduce variation later
sigmaks = np.ones(n)*0.3e-15 #fixed gaussian width
#Time domain
t=np.linspace(-10e-15,10e-15,x)
#Frequency Domain
wrange = np.array([1-1e-3,1+1e-3])*w_cen #Domain centered around central frequency
w=np.arange(wrange[0],wrange[1],(wrange[1]-wrange[0])/x)

fig, axs = plt.subplots(3, 1)

efieldtmpt = time_Efield(Aks,wks,phiks,tks,sigmaks,t)
tfield = efieldtmpt.time_field()
axs[0].plot(t,np.real(tfield),label='Real')
axs[0].plot(t,np.imag(tfield),label='Imag')
axs[0].legend()

efieldtmpf = freq_Efield(Aks,wks,phiks,tks,sigmaks,w)
ffield = efieldtmpf.freq_field()
axs[1].plot(w,np.real(ffield),label='Real')
axs[1].plot(w,np.imag(ffield),label='Imag')
axs[1].legend()

t_fix = np.fft.ifftshift(t)
efieldtmpt_fix = time_Efield(Aks,wks,phiks,tks,sigmaks,t_fix)
tfield_fix = efieldtmpt_fix.time_field()
ffield_fix = np.fft.fft(tfield_fix)
axs[2].plot(w,np.real(ffield_fix),label='Real')
axs[2].plot(w,np.imag(ffield_fix),label='Imag')
axs[2].legend()

fig1, axs1 = plt.subplots(2, 1)
axs1[0].plot(w,np.abs(ffield),label='Magnitude')
axs1[1].plot(w,np.abs(ffield_fix),label='Magnitude')

