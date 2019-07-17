#coding=utf-8
"""
filename:       oneD_phase_retrieval.py
Description:
Author:         Sharon Dengjie Xiao
IDE:            PyCharm
Change:         2019/6/30  上午11:44    Sharon Dengjie Xiao        Create


"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test = pd.read_csv("test.csv").iloc[:,1]
# freqs = pd.read_csv("test_fft.csv").iloc[:,1]
# freqs = pd.read_csv("FreqIntensity.csv").iloc[1,range(0,1000,10)]
# freqs = pd.read_csv("FreqElectric.csv").iloc[1,range(0,1000,1)]*10000000000000000000
freqs = pd.read_csv("FreqElectric.csv").iloc[1,range(0,1000,1)]
# freqs = np.abs(np.fft.fft(freqs))


np.random.seed(5)
guess =freqs * np.exp(1j * np.random.rand(len(freqs)) * 2 * np.pi)

prev = None

print('guess', guess)

print(prev)
it = 3

for i in range(it):

    update_freqint = freqs * np.exp(1j * np.angle(guess))




    timeint = np.fft.ifft(update_freqint)
    prev = timeint #store to graph later

    real_timeint = np.real(timeint)
    # real_timeint = np.abs(np.real(timeint))

    freqint = np.fft.fft(real_timeint)


    guess = np.exp(1j * np.angle(freqint))
    # print("guss",guess)

    #print('real', real)

    #print('new guess', guess)


    #print('len prev', len(prev))



    if i % 1 == 0:
        print(i)
        print('prev', prev.size)
        fig, ax = plt.subplots()
        ax.plot(np.real(prev),c='g')
        #ax.plot(np.imag(prev))
        # ax.plot(test,c='b')
        # plt.plot(intensity)
        plt.savefig("d"+str(i) + '.jpg')

# plt.plot(firstrow, time)
# plt.savefig('intensity_sol.jpg')





