# Authored by Mathuranathan Viswanathan 
# How to generate AWGN noise in Matlab/Octave by Mathuranathan Viswanathan 
# is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0  International License.
import numpy as np
import math as math
import matplotlib.pyplot as plt
import pylab as m

def awgn(x,SNR_db):
    L = len(x)
    SNR_lin = 10**((SNR_db)/10)
    
    if np.iscomplexobj(x) == True:
        E = m.zeros(L, 'complex')
        for i in range(L):
            E[i] = np.abs(x[i])**2
        Esym = np.average(E)
        N0 = Esym/SNR_db
        noiseSigma = np.sqrt(N0/2)
        n = noiseSigma * (np.random.normal(0,1,L) + 1j*np.random.normal(0,1,L))
    
    elif np.iscomplexobj(x) == False:
        E = np.zeros(L)
        for i in range(L):
            E[i] = np.abs(x[i])**2
        Esym = np.average(E)
        N0 = Esym/SNR_db
        noiseSigma = np.sqrt(N0)
        n = noiseSigma * np.random.normal(0,1,L)
    
    y = x + n
    return y


    