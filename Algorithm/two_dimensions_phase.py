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
import scipy as misc
from imageio import imwrite
from math import pi
import matplotlib.pyplot as plt
import cv2
import pandas as pd

#save_path = "/User/users/Desktop/Image-Reconstruction/save8-fel/"
#pic_path = r"C:\Users\user\Desktop\test_phase\"

# Read in source image
# source = mlimr("einstein.bmp")

def crop_image(img, pad_len, source_width, source_height):
    #img is image data
    #tol is tolerance
    if source_width > source_height:
        s = source_width
    else:
        s= source_height

    x1, y1 = pad_len + s , pad_len + s
    x0, y0 = pad_len, pad_len + 1

    cropped = img[x0 : x1, y0 : y1]
    return cropped

img = cv2.imread("hoovertower.jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
source_height, source_width = img.shape[:2]

source = img

if source_height > source_width:
    diff = source_height - source_width
    source = cv2.copyMakeBorder(img, 0, 0, 0, diff, cv2.BORDER_CONSTANT,
                                value = [0, 0, 0])

if source_width > source_height:
    diff = source_width - source_height
    source = cv2.copyMakeBorder(img, 0, diff, 0, 0, cv2.BORDER_CONSTANT,
                                value = [0, 0, 0])




# Pad image to simulate oversampling
pad_len = len(source)
print("pad_len",pad_len)
padded = np.pad(source, ((pad_len, pad_len), (pad_len, pad_len)), 'constant',
                constant_values=((0, 0), (0, 0)))

ft = fft.fft2(padded)
#save start phase
df_inv = pd.DataFrame(np.angle(ft))
df_inv.to_csv( str('start') + "phase.csv")


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
        res = crop_image(prev, pad_len, source_width, source_height)
        plt.imshow(res)
        plt.axis("off")
        plt.savefig(str(s) + '.jpg')  # , prev  ".bmp"

        df_inv = pd.DataFrame(np.angle(guess))
        df_inv.to_csv(str(s) + "phase.csv")
        # break #test save csv
        print(s)

print("finish")


