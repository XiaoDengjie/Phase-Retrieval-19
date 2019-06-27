#coding=utf-8
"""
filename:       test_phase.py
Description:
Author:         Dengjie Xiao
IDE:            PyCharm
Change:         2019/6/17  上午12:24    Dengjie Xiao        Create


"""
import numpy as np
import numpy.fft as fft
from matplotlib.pyplot import imread as mlimr
import scipy.misc as misc
from imageio import imwrite
from math import pi
import matplotlib.pyplot as plt
import cv2
import pandas as pd

save_path = "/Users/xiaodengjie/Desktop/Phase Retrieval Code/Image-Reconstruction/save/save9-XTCAV/"
pic_path = "/Users/xiaodengjie/Desktop/Phase Retrieval Code/Image-Reconstruction/pic/"

# Read in source image
# source = mlimr("einstein.bmp")

source = mlimr(pic_path+"fel.png")[0:218,0:218]
source = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)

# Pad image to simulate oversampling
pad_len = len(source)
print("pad_len",pad_len)
padded = np.pad(source, ((pad_len, pad_len), (pad_len, pad_len)), 'constant',
                constant_values=((0, 0), (0, 0)))

ft = fft.fft2(padded)
#save start phase
df_inv = pd.DataFrame(np.angle(ft))
df_inv.to_csv(save_path + str('start') + "phase.csv")


# simulate diffraction pattern
diffract = np.abs(ft)

l = len(padded)

# keep track of where the image is vs the padding
mask = np.ones((pad_len + 2, pad_len + 2))
mask = np.pad(mask, ((pad_len - 1, pad_len - 1), (pad_len - 1, pad_len - 1)), 'constant',
              constant_values=((0, 0), (0, 0)))



# Initial guess using random phase info
guess = diffract * np.exp(1j * np.random.rand(l, l) * 2 * pi)

# number of iterations
r = 801

# step size parameter
beta = 0.8

# previous result
prev = None
for s in range(0, r):
    # apply fourier domain constraints
    update = diffract * np.exp(1j * np.angle(guess))

    inv = fft.ifft2(update)

    # #=======================
    # #    inv =np.abs(inv)
    #inv_imag = np.imag(inv)
    #print("inv_imag", inv_imag)

# write fft process to txt
    # with open(save_path + "phase.txt","a") as  rt:
    #     rt.write(str(inv))

    # print('inv',inv)
    inv = np.real(inv)


    if prev is None:
        prev = inv

    # apply real-space constraints
    temp = inv
    for i in range(0, l):
        for j in range(0, l):
            # image region must be positive
            if inv[i, j] < 0 and mask[i, j] == 1:
                inv[i, j] = prev[i, j] - beta * inv[i, j]
            # push support region intensity toward zero
            if mask[i, j] == 0:
                inv[i, j] = prev[i, j] - beta * inv[i, j]

    prev = temp  # np.uint8(temp)

    guess = fft.fft2(inv)


    # save an image of the progress
    if s % 100 == 0:
        # imwrite
        plt.imshow(prev)
        plt.savefig(save_path + str(s) + '.jpg')  # , prev  ".bmp"

        df_inv = pd.DataFrame(np.angle(guess))
        df_inv.to_csv(save_path + str(s) + "phase.csv")
        # break #test save csv
        print(s)

print("finish")


