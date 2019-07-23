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
tks = np.arange(0,n)*1e-15 - 0.5e-15 #Will probably introduce variation later
sigmaks = np.ones(n)*0.3e-15 #fixed gaussian width
#Time domain
t=np.linspace(-10e-15,10e-15,x)
#Frequency Domain
wrange = np.array([1-1e-3,1+1e-3])*w_cen #Domain centered around central frequency
w=np.arange(wrange[0],wrange[1],(wrange[1]-wrange[0])/(x))

#Generate Data
efieldtmpt = time_Efield(Aks,wks,phiks,tks,sigmaks,t)
tfield = efieldtmpt.time_field()
efieldtmpf = freq_Efield(Aks,wks,phiks,tks,sigmaks,w)
ffield = efieldtmpf.freq_field()

t_shift = np.fft.ifftshift(t)
efieldtmpt_shift = time_Efield(Aks,wks,phiks,tks,sigmaks,t_shift)
tfield_shift = efieldtmpt_shift.time_field()

ffield_fft = np.fft.fft(tfield_shift)

#Plot general information to check if fft and analytical are same
#'''
fig, axs = plt.subplots(3, 1)
axs[0].set_title("Electric Field in Time")
axs[0].plot(t,np.real(tfield),label='Real')
axs[0].plot(t,np.imag(tfield),label='Imag')
axs[0].legend()

axs[1].set_title("Electric Field in Frequency: Analytical")
axs[1].plot(w,np.real(ffield),label='Real')
axs[1].plot(w,np.imag(ffield),label='Imag')
axs[1].legend()

axs[2].set_title("Electric Field in Frequency: Fast Fourier Transform")
axs[2].plot(np.real(ffield_fft),label='Real')
axs[2].plot(np.imag(ffield_fft),label='Imag')
axs[2].legend()


fig1, axs1 = plt.subplots(1, 2)
axs1[0].plot(w,np.abs(ffield),label='Magnitude')
axs1[0].legend()
axs1[1].plot(w,np.abs(ffield_fft),label='Magnitude')
axs1[1].legend()
#'''

TimeElectricAmp = np.abs(tfield)
FreqElectricAmp = np.abs(ffield_fft)

fig1, axs1 = plt.subplots(1, 2)
axs1[0].set_title('Electric Field Time Domain Amplitude')
axs1[0].plot(TimeElectricAmp,label='Magnitude')
axs1[1].set_title('Electric Field Frequency Domain Amplitude')
axs1[1].plot(FreqElectricAmp,label='Magnitude')


def GSFreq(iterations,TimeAmp,FreqAmp,seed):
    #set seed if you want to test same initial conditions
    np.random.seed(seed)
    #generate random inital phase
    phase_time = 2*np.pi*np.random.random(len(FreqAmp))
    for i in range(iterations):
        #Multiply phase with initial amplitude in time space
        Et = np.exp(1j*phase_time)*TimeAmp
        #Go to frequency space via fourier transform
        Guess_Freq = np.fft.fft(np.fft.ifftshift(Et))
        #Take new phase in frequency space
        phase_freq = np.angle(Guess_Freq)
        #Multiply phase with initial amplitude in frequency space
        Ew = np.exp(1j*phase_freq)*FreqAmp
        #Perform inverse fourier transform
        Guess_Time = np.fft.fftshift(np.fft.ifft(Ew))
        #Get angle of phase
        phase_time = np.angle(Guess_Time) 
    return Et, Ew

#Et, Ew = GSFreq(100,TimeElectricAmp,FreqElectricAmp,1)
'''
#Good way to graph an iteration of GS against actual functions
fig2, axs2 = plt.subplots(2, 2)
axs2[0,0].set_title("GS Data")
axs2[0,0].plot(np.real(Et),label='Real')
axs2[0,0].plot(np.imag(Et),label='Imag')
axs2[0,0].legend()
axs2[1,0].plot(np.real(Ew),label='Real')
axs2[1,0].plot(np.imag(Ew),label='Imag')
axs2[1,0].legend()
axs2[0,1].set_title("Simulated Data")
axs2[0,1].plot(np.real(tfield),label='Real')
axs2[0,1].plot(np.imag(tfield),label='Imag')
axs2[0,1].legend()
axs2[1,1].plot(np.real(ffield_fft),label='Real')
axs2[1,1].plot(np.imag(ffield_fft),label='Imag')
axs2[1,1].legend()
'''

#Define Normalization function
def Normalize(actual,guess):
    numer = np.abs(guess - actual)
    denom = np.sqrt(np.abs(guess)**2 + np.abs(actual)**2)
    Normal = numer/denom
    return Normal

#Generate lots of data points to observe best initial condition
z = 100
All_TimeDiff = np.zeros(z)
All_FreqDiff = np.zeros(z)
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

print(All_TimeDiff[90])

#Check lowest seed number: it is 91. 90,15,30 all good
np.argmin(All_TimeDiff)
Et, Ew = GSFreq(100,TimeElectricAmp,FreqElectricAmp,23)
fig2, axs2 = plt.subplots(2, 2)
axs2[0,0].set_title("GS Data")
axs2[0,0].plot(np.real(Et),label='Real')
axs2[0,0].plot(np.imag(Et),label='Imag')
axs2[0,0].legend()
axs2[1,0].plot(np.real(Ew),label='Real')
axs2[1,0].plot(np.imag(Ew),label='Imag')
axs2[1,0].legend()
axs2[0,1].set_title("Simulated Data")
axs2[0,1].plot(np.real(tfield),label='Real')
axs2[0,1].plot(np.imag(tfield),label='Imag')
axs2[0,1].legend()
axs2[1,1].plot(np.real(ffield_fft),label='Real')
axs2[1,1].plot(np.imag(ffield_fft),label='Imag')
axs2[1,1].legend()

#Observe freq and time error per iteration
x = 20
IterErrorFreq = np.zeros(x)
IterErrorTime = np.zeros(x)
for i in range(1,x):
    Et, Ew = GSFreq(i,TimeElectricAmp,FreqElectricAmp,15)
    TimeDifference = np.linalg.norm(np.abs(Et-tfield)**2)
    FreqDifference = np.linalg.norm(np.abs(Ew-ffield_fft)**2)
    IterErrorTime[i] = TimeDifference
    IterErrorFreq[i] = FreqDifference

fig4, axs4 = plt.subplots(2, 1)
axs4[0].plot(IterErrorTime)
axs4[1].plot(IterErrorFreq)

#Compare two seeds that both converge to basically same, correct solution
Et90, Ew90 = GSFreq(100,TimeElectricAmp,FreqElectricAmp,90)
Et91, Ew91 = GSFreq(100,TimeElectricAmp,FreqElectricAmp,91)
DiffSolu = np.linalg.norm(np.abs(Et90-Et91)**2)
print(DiffSolu)

np.random.seed(90)
phase_time90 = np.exp(1j*2*np.pi*np.random.random(len(FreqElectricAmp)))
np.random.seed(91)
phase_time91 = np.exp(1j*2*np.pi*np.random.random(len(FreqElectricAmp)))

plt.figure()
plt.plot((phase_time90))
plt.plot(phase_time91)
