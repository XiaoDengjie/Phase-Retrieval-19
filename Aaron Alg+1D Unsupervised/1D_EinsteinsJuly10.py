import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load in intensities
TimeIntensity = pd.read_csv("July-10\TimeIntensity.csv")
TimeI = np.array(TimeIntensity.iloc[0,:])

FreqIntensity = pd.read_csv("July-10\FreqIntensity.csv")
FreqI = np.array(FreqIntensity.iloc[0,:])

#Fourier pair by data
TimeE = np.sqrt(TimeI)
FreqE = np.sqrt(FreqI)

#Fourier pair by numpy
TimeE = np.sqrt(TimeI)
FreqE = np.abs(np.fft.fft(TimeE))

'''

Useful for visualization of function converging to fourier pair
if (i % 50) == 0:
    plt.figure(i)
    plt.plot(np.abs(Time_ifft))
'''



def GS(iterations,TimeE,FreqE):
    phase_time = 2*np.pi*np.random.random(len(TimeE))
    n = iterations
    for i in range(n):
        Et = np.exp(1j*phase_time)*TimeE
        Guess_Freq = np.fft.fft(Et)
        phase_freq = np.angle(Guess_Freq)
        Ew = np.exp(1j*phase_freq)*FreqE
        Guess_Time = np.fft.ifft(Ew)
        phase_time = np.angle(Guess_Time)
    return Et, Ew

Et,Ew = GS(100,TimeE,FreqE)
#Define M as the loss function
M = Et - TimeE

plt.figure()
plt.plot(np.imag(Et))