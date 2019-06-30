#coding=utf-8
"""
filename:       simdata_testphase.py
Description:
Author:         Sharon Huang
IDE:            PyCharm
Change:         2019/6/27  上午12:24    Sharon Huang        Create


"""
import numpy as np
import numpy.fft as fft
from matplotlib.pyplot import imread as mlimr
import scipy as misc
from imageio import imwrite
from math import pi
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import csv
import numpy
import pandas
from pathlib import Path

#save_path = "/User/users/Desktop/Image-Reconstruction/save8-fel/"
#pic_path = r"C:\Users\user\Desktop\test_phase\"

# Read in source image
# source = mlimr("einstein.bmp")
n = 100

times = pandas.read_csv("Time.csv")
time = times.iloc[0,:]

data = pandas.read_csv("TimeIntensity.csv")
firstrow = data.iloc[0,:]
#print(len(firstrow))
source = []
for i in range(1, len(firstrow), 10):
     source.append(firstrow.tolist()[i])
#print(source)
#int_data = pd.DataFrame(int_data)



freq_data = pandas.read_csv("FreqIntensity.csv")
freq_firstrow = freq_data.iloc[0,:]
print(freq_firstrow)
freqs = []
for i in range(1, len(freq_firstrow) - 1, 10):
    freqs.append(freq_firstrow.tolist()[i])
print(freqs[1:10])




guess = np.exp(1j * np.random.rand(100) * 2 * pi)* freqs*10**30
prev = None

print('guess', guess)

print(prev)
it = 2

for i in range(it):


    new_phase = fft.ifft(guess)   #guess of phase graph
    print('new_phase', new_phase)

    real = np.real(new_phase)
    print('real', real)
    prev = real # extract prev to plot phase

    guess = fft.fft(real)
    print('new guess', guess)


    #print('len prev', len(prev))



    if i % 1 == 0:
        print(i)
        plt.plot(prev)
        plt.savefig(str(i) + '.jpg')

plt.plot(firstrow, time)
plt.savefig('intensity_sol.jpg')


