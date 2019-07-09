#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:01:36 2019

@author: gzhou
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.arange(-100,100,0.01)
#x=np.fft.ifftshift(x)
y=np.exp(-x**2)

plt.plot(x,y,'.-')

z=np.fft.fftshift(np.fft.fft(y))
plt.subplot(3,1,1)
plt.plot(x,z,'.-')
plt.xlim([-2,2])
plt.subplot(3,1,2)
plt.plot(x,np.angle(z),'.-')
plt.xlim([-2,2])
plt.subplot(3,1,3)
plt.plot(x,np.abs(z)**2,'.-')
plt.xlim([-2,2])

x=np.arange(-100,100,0.01)
x=np.fft.ifftshift(x)
y=np.exp(-x**2)

plt.plot(x,y,'.-')

z=np.fft.fft(y)
plt.subplot(3,1,1)
plt.plot(x,z,'.-')
plt.xlim([-2,2])
plt.subplot(3,1,2)
plt.plot(x,np.angle(z),'.-')
plt.xlim([-2,2])
plt.subplot(3,1,3)
plt.plot(x,np.abs(z)**2,'.-')
plt.xlim([-2,2])














N=len(y)
z1=np.zeros([N,1])
ks= np.arange(0,N)
for i in range(N):
    exppart = np.exp(-1j*2*np.pi/N*i*ks)
    z[i] = np.sum(exppart*y)
    




y1=np.fft.ifft(np.fft.ifftshift(z))
plt.plot(x,y1)
plt.plot(x,np.abs(y1-y))


z=np.fft.fft(np.fft.ifftshift(y))
plt.subplot(3,1,1)
plt.plot(x,z,'.-')
plt.xlim([-2,2])
plt.subplot(3,1,2)
plt.plot(x,np.angle(z),'.-')
plt.xlim([-2,2])
plt.subplot(3,1,3)
plt.plot(x,np.abs(z)**2,'.-')
plt.xlim([-2,2])


