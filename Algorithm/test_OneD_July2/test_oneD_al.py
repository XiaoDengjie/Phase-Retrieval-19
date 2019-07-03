"""


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test = pd.read_csv("test.csv").iloc[:,1]
freqs = pd.read_csv("c_fft.csv").iloc[:,1]
# print(freqs)
guess = np.exp(1j * np.random.rand(100) * 2 * np.pi)
# plt.plot(guess)
# plt.plot(freqs,c='g')
# plt.plot(test , c='red')
# plt.show()


prev = None

print('guess', guess)

print(prev)
it = 2

for i in range(it):


    update_freqint = [freqs[i] * guess[i] for i in range(len(guess))]   #guess of phase graph
    #print('new_phase', new_phase)
    plt.plot(update_freqint)
    plt.show()

    timeint = np.fft.ifft(update_freqint)
    prev = timeint #store to graph later

    real_timeint = np.real(timeint)

    freqint = np.fft.fft(real_timeint)


    guess = np.exp(1j * np.angle(freqint))
    print(guess)

    #print('real', real)

    #print('new guess', guess)


    #print('len prev', len(prev))



    if i % 100 == 0:
        print(i)
        print('prev', prev.size)
        plt.plot(np.real(prev))
        # plt.plot(intensity)
        plt.savefig(str(i) + '.jpg')

# plt.plot(firstrow, time)
# plt.savefig('intensity_sol.jpg')


