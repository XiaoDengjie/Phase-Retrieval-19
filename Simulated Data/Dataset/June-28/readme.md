This data was generated with normalized constants except for random Phis.

phiks = np.random.random(n)*2*np.pi

Aks = np.ones(n)
wks = np.ones(n)*w_cen
tks = np.arange(1,n+1)*1e-15
sigmaks = np.ones(n)*0.3e-15
t=np.arange(0,(n+1)*1e-15,12e-18)

wrange = np.array([1-1e-3,1+1e-3])*w_cen
w=np.arange(wrange[0],wrange[1],(wrange[1]-wrange[0])/1000)

10 total datasets

Included time intensity plots to compare to generated IFFT plots