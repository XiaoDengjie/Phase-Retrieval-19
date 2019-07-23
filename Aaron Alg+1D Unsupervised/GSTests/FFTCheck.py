# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:31:58 2019

@author: aholm
"""
#Checks if Analytical and FFT function produce same result
import numpy as np
from matplotlib import pyplot as plt
from Generator import t, w, tfield, ffield, ffield_fft

#Plot Time E field, Analytic freq, and FFT freq
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

#Plot amplitudes of analytical and FFT frequency
fig1, axs1 = plt.subplots(1, 2)
axs1[0].plot(w,np.abs(ffield),label='Magnitude')
axs1[0].legend()
axs1[1].plot(w,np.abs(ffield_fft),label='Magnitude')
axs1[1].legend()

#Plot amplitudes of time and freq space using FFT function
TimeElectricAmp = np.abs(tfield)
FreqElectricAmp = np.abs(ffield_fft)

fig2, axs2 = plt.subplots(1, 2)
axs2[0].set_title('Electric Field Time Domain Amplitude')
axs2[0].plot(TimeElectricAmp,label='Magnitude')
axs2[1].set_title('Electric Field Frequency Domain Amplitude')
axs2[1].plot(FreqElectricAmp,label='Magnitude')