import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class time_Efield(object):
    #Generate the electric field in the time domain.
    def __init__(self, Aks,wks,phiks,tks,sigmaks,t):
        #Amplitude of each gaussian
        self.Aks = Aks
        #Frequencies at which each wave is generated at
        self.wks = wks
        #Phase term. This is what we will be attempting to retrieve through ML
        self.phiks = phiks
        #Time in which pulse occurred.
        self.tks = tks
        #Width of gaussian
        self.sigmaks = sigmaks
        #Number of pulses
        self.modes_num = len(Aks)
        #Time domain
        self.t = t
    def time_single(self, Ak,wk,phik,tk,sigmak):
        #note: t should be an numpy array
        phase = wk*self.t+phik
        gaussian = -((self.t-tk)**2)/2/sigmak**2
        Efield_single = Ak*np.exp(1j*phase)*np.exp(gaussian)
        return Efield_single
    def time_field(self):
        efield = np.zeros([self.modes_num,len(self.t)],dtype=complex)
        for i in range(self.modes_num):
            efield[i,:] = self.time_single(self.Aks[i],self.wks[i],\
                  self.phiks[i],self.tks[i],self.sigmaks[i])
        return np.sum(efield,axis=0)

#Constants
e_charge = 1.602176565e-19
h_plank = 6.62607004e-34
c_speed = 299792458
x = 1000 #Number of points to test in frequeny and time domain

#Inputs
ph_en = 8300 #in the uint of eV
w_cen =  ph_en * e_charge /h_plank *2* np.pi #frequency of radiation
n = 2 #Number of modes
np.random.seed(1) #Set seed if want to create reproducable results

#Parameters
Aks = np.ones(n) #np.random.rayleigh(size=n) is true variation
wks = np.ones(n)*w_cen #np.random.normal(w_cen,10**-4*w_cen,n) true variation
phiks = np.random.random(n)*2*np.pi #random values between 0 and 2pi
tks = np.arange(1,n+1)*1e-15 #Will probably introduce variation later
sigmaks = np.ones(n)*0.3e-15 #fixed gaussian width
#Time domain
t=np.linspace(-5e-15,(n+1)*1e-15 + 5e-15,x)
#Frequency Domain
wrange = np.array([1-1e-3,1+1e-3])*w_cen #Domain centered around central frequency
w=np.arange(wrange[0],wrange[1],(wrange[1]-wrange[0])/x)






#Generate electric fields in time and frequency domains
efieldtmpt = time_Efield(Aks,wks,phiks,tks,sigmaks,t)
TimeE = efieldtmpt.time_field()
FreqE = np.fft.fft(TimeE)

TimeAmp = np.abs(TimeE)
FreqAmp = np.abs(FreqE)

def GS(TimeE,FreqE):
    phase_time = 2*np.pi*np.random.random(len(TimeE))
    n = 0
    FreqError = 10
    TimeError = 10
    MinFreqError = 10**-0
    MinTimeError = 10**-1
    while FreqError >= MinFreqError and TimeError >= MinTimeError and n<= 300:
        Et = np.exp(1j*phase_time)*TimeE
        Guess_Freq = np.fft.fft(Et)
        FreqError = np.linalg.norm(Guess_Freq - FreqE)
        DiffError = np.linalg.norm(np.abs(Guess_Freq)-FreqAmp)
        print(DiffError)
        phase_freq = np.angle(Guess_Freq)
        Ew = np.exp(1j*phase_freq)*FreqE
        Guess_Time = np.fft.ifft(Ew)
        TimeError = np.linalg.norm(Guess_Time - TimeE)
        phase_time = np.angle(Guess_Time)
        n+= 1
    return Et, Ew, n


a,b,c = GS(TimeAmp,FreqAmp)

z = 100
AllTime=np.zeros((1000,z),dtype=complex)
AllFreq=np.zeros((1000,z),dtype=complex)
AllIterations = np.zeros(z)
for i in range(z):
    np.random.seed(i)
    ElectricTime, ElectricFreq, Iterations = GS(TimeAmp,FreqAmp)
    AllTime[:,i] = ElectricTime
    AllFreq[:,i] = ElectricFreq
    AllIterations[i] = Iterations


plt.plot(AllIterations)

NormDiff = np.zeros(z)

for i in range(z):
    Difference = AllTime[:,i] - TimeE
    AbsDiff = np.abs(Difference)**2
    Norm = np.linalg.norm(AbsDiff)
    NormDiff[i] = Norm

plt.plot(NormDiff)

print(TimeE)















