# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:23:44 2019

@author: aholm
"""
import numpy as np

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