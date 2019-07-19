# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:35:55 2019

@author: aholm
"""
import numpy as np
from matplotlib import pyplot as plt

N = 100
x = np.linspace(-10,10,N)
y = np.exp(-(x)**2)

fig, axs = plt.subplots(4, 1)
axs[0].plot(x,y)

fft_y = np.fft.fft(y)

axs[1].plot(x,np.real(np.fft.fftshift(fft_y)),label="Real")
axs[1].plot(x,np.imag(np.fft.fftshift(fft_y)),label="Imag")
axs[1].legend()

x_shift = np.fft.ifftshift(x)
y_shift = np.exp(-(x_shift)**2)
fft_y_shift = np.fft.fft(y_shift)

axs[2].plot(x_shift,np.real(fft_y_shift),label="Real")
axs[2].plot(x_shift,np.imag(fft_y_shift),label="Imag")
axs[2].legend()

N = 101
x_fix = np.fft.ifftshift(np.linspace(-10,10,N))
y_fix = np.exp(-(x_fix)**2)
fft_y_fix = np.fft.fft(y_fix)
axs[3].plot(x_fix,np.real(fft_y_fix),label="Real")
axs[3].plot(x_fix,np.imag(fft_y_fix),label="Imag")
axs[3].legend()


