import numpy as np
from math import *
import matplotlib.pyplot as plt
from TheSignals import *
import pylab as m
from awgn import *


def babybaby2(filterlength, stepsize, setSNR):
    ################## CALLING SIGNALS FROM OTHER SCRIPT############################
    data = papersignals('fixed')
    train = data['train']
    noisy = awgn(train, setSNR)
    pulse = data['pulse']
    t = data['time']
    
    ################## ADAPTIVE FILTER PARAMETERS ##################################
    L = filterlength
    mu =  stepsize #4.9e-4 #10**-5
    sample_num = len(train)
    delay = 5 #number of sample delays
    
    ###############################################################################
    
    simulation_time = 50
    
    AMSE = np.zeros(sample_num-L) 
    for p in range(simulation_time):
        ######################## LEAST MEAN SQUARE FILTER###############################
        
        ####################WORK WITH REAL/IMAG SIGNALS#################################
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
        err_sq = np.zeros(sample_num-L)
        for i in range(0,sample_num-L):
            x_n = x_delta[i:i+L]                #section of the recieved signal
            output[i] = np.dot(x_n,w)           #section * filter weights
            e = x[i+L] - output[i]              #error for lms
            err_sq[i] = np.abs(e**2) 
            w = w + 2*mu*e*x_n                  #update filter weights
         
            if i == sample_num-L-1:             #store the last 
                optimal += w
        filtered = np.concatenate([output, np.zeros(L)])#pad to match timeline
        
        MSE = np.zeros(len(err_sq))
        for i in range(len(err_sq)):
            if i == 1:
                MSE[i] = err_sq[i]
            elif i == 2:
                MSE[i] = (err_sq[i] + err_sq[i-1])/2
            elif i == 3:
                MSE[i] = (err_sq[i] + err_sq[i-1] +err_sq[i-2])/3
            else:
                MSE[i] = (err_sq[i] + err_sq[i-1] + err_sq[i-2] +err_sq[i-3])/4
        
        AMSE = AMSE + MSE
    
    AMSE =AMSE/50
    fs = 1280e6
    convLim = np.average(AMSE[len(pulse)-200:len(pulse)])
    LL = L+1
    av_init = np.average(AMSE[L+1:L+1+20])
    
    xaxis = np.linspace(1, len(AMSE), len(AMSE))
    xaxis = xaxis/fs
    
    for i in range(len(pulse)):
        if av_init > convLim:
            av_init = np.average(AMSE[i + L+1:i + L+1+20])
        else:
            convtime = xaxis[i]
            break

    return convtime


L_line = np.arange(2, 102, 49)
mu_line = np.linspace(0.00001, 0.00049, len(L_line))

matrix = np.zeros(shape=(len(L_line),len(L_line)))
for i in range(len(L_line)):
    trap = L_line[i]
    row = np.zeros(len(mu_line))
    print trap
    for j in range(len(mu_line)):
        store = babybaby2(trap, mu_line[j], 10)
        if np.isnan(store) == True:
            row[j] = row[j-1]
        else:
            row[j] = store
        print store
    matrix[i] = row
 
x, y = np.meshgrid(mu_line, L_line)
intensity = np.array(matrix)
plt.pcolormesh(x, y, intensity)
plt.title('Convergance time (s) - Fixed Frequency pulse with Input SNR 10dB')
plt.xlabel('Step size $\mu$')
plt.ylabel('Filter length -  L')
plt.gca().invert_yaxis()
plt.colorbar() #show the bar on the side
# plt.show() #show graph
plt.savefig('10dbfixed.png')