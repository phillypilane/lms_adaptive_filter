import numpy as np
from math import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from TheSignals import *
import pylab as m
from awgn import *

def babybaby(filterlength, stepsize, setSNR):
    data = papersignals('fixed')
    train = data['train']
    noisy = awgn(train, setSNR)
    pulse = data['pulse']
    t = data['time']
    
    L = filterlength
    mu =  stepsize #4.9e-4 #10**-5
    sample_num = len(train)
    delay = 5 #number of sample delays
    
    x = np.real(noisy)                      #recieved signal
    d = np.real(train)                      #desired
    w = np.zeros(L)                         #filter weights
     
    output = np.zeros(sample_num-L)
     
    #delaying the signal
    x_delta = np.roll(x, delay)             #delaying the signal
    for i in range(delay):                  #making the first 5 zeros
        x_delta[i] = 0.0
    x_delta = np.concatenate([np.zeros(L-1), x_delta]) #adding zeros to front
     
    optimal = np.empty(L) #optimal weights
    for i in range(0,sample_num-L):
        x_n = x_delta[i:i+L]                      #section of the recieved signal
        output[i] = np.dot(x_n,w)           #section * filter weights
        e = x[i+L] - output[i]              #error for lms
        w = w + 2*mu*e*x_n                  #update filter weights
     
        if i == sample_num-L-1:             #store the last 
            optimal += w
     
    filtered = np.concatenate([output, np.zeros(L)])#pad to match timeline
    
    f_d = np.convolve(optimal, d)           #convolution of optimal filter weights and desired (no noise)
    just_noise = x - d                      #subtraction of clean from noisy to get just noise
    f_n = np.convolve(just_noise, optimal)       #convolution of optimal filter weights and noise
    
    ####calculating powers for snr calcs
    Ps = np.zeros(len(f_d))
    Pn = np.zeros(len(f_d))
    for i in range(len(f_d)):
        Ps[i] = np.abs(f_d[i])*np.abs(f_d[i])
        Pn[i] = np.abs(f_n[i])*np.abs(f_n[i])

    calcSNR = np.average(Ps)/np.average(Pn) #SNR in linear
    calcSNR = 10 * np.log10(calcSNR)        #SNR in dB
    
    gain = calcSNR - setSNR
    
    return gain

##heatmap generation

L_line = np.arange(2, 102, 1)
mu_line = np.linspace(0.00001, 0.00049, len(L_line))

matrix = np.zeros(shape=(len(L_line),len(L_line)))
for i in range(len(L_line)):
    trap = L_line[i]
    row = np.zeros(len(mu_line))
    print trap
    for j in range(len(mu_line)):
        store = babybaby(trap, mu_line[j], 10)
    if np.isnan(store) == True:
        row[j] = row[j-1]
        else:
        row[j] = store
    matrix[i] = row
 
x, y = np.meshgrid(mu_line, L_line)
intensity = np.array(matrix)
plt.pcolormesh(x, y, intensity)
plt.title('SNR Gain [dB] - Input SNR 10dB')
plt.xlabel('Step size $\mu$')
plt.ylabel('Filter length -  L')
plt.gca().invert_yaxis()
plt.colorbar() #show the bar on the side
plt.show() #show graph