import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load in intensities
TimeElectric = pd.read_csv("July-15\ElectricFieldTime.csv")
TimeE = np.array(TimeElectric.iloc[0,:])

FreqElectric = pd.read_csv("July-15\ElectricFieldFreq.csv")
FreqE = np.array(FreqElectric.iloc[0,:])


plt.figure()
plt.plot(TimeE)
plt.figure()
plt.plot(FreqE)

'''

Useful for visualization of function converging to fourier pair
if (i % 50) == 0:
    plt.figure(i)
    plt.plot(np.abs(Time_ifft))
'''


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
'''