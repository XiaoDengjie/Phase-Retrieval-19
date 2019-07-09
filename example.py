#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:18:28 2019

@author: gzhou
"""

import numpy as np
import matplotlib.pyplot as plt

class time_Efield(object):

    def __init__(self, Aks,wks,phiks,tks,sigmaks,t):
        self.Aks = Aks
        self.wks = wks
        self.phiks = phiks
        self.tks = tks
        self.sigmaks = sigmaks
        self.modes_num = len(Aks)
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

ph_en = 8300 #in the uint of eV
e_charge = 1.602176565e-19
h_plank = 6.62607004e-34
c_speed = 299792458  
w_cen =  ph_en * e_charge /h_plank *2* np.pi
n = 2
Aks = np.ones(n)
# np.random.random(n)
wks = np.ones(n)* w_cen
phiks = np.random.random(n)*2*np.pi
tks = np.arange(-20,20,20)*1e-15
sigmaks = np.ones(n)*3e-15
t=np.arange(-100e-15,100e-15,1e-18)
efieldtmpt = time_Efield(Aks,wks,phiks,tks,sigmaks,t)
tfield = efieldtmpt.time_field()
plt.figure(1)
plt.plot(t,abs(tfield)**2)


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
    
wrange = np.array([1-1e-3,1+1e-3])*w_cen   
w=np.arange(wrange[0],wrange[1],(wrange[1]-wrange[0])/10000)
efieldtmpf = freq_Efield(Aks,wks,phiks,tks,sigmaks,w)
ffield = efieldtmpf.freq_field()
plt.figure(2)
plt.plot(w,abs(ffield)**2)  


  
plt.figure(3)
plt.plot(w,ffield) 
plt.figure(4)     
plt.plot(np.fft.fftshift(np.fft.fft(tfield)))


 

plt.figure(5)     
pretfield = np.fft.ifftshift(tfield)
fft_ffield= np.fft.fftshift(np.fft.fft(pretfield))
plt.plot(fft_ffield)
    
    
    
    
    
    
    
    
    
    
    
    
    
    