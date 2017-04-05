import numpy as np
from math import *
import matplotlib.pyplot as plt
from TheSignals import *
import pylab as m
from awgn import *

################## CALLING SIGNALS FROM OTHER SCRIPT############################
data = papersignals('bar')
train = data['train']
noisy = awgn(train, 3)
pulse = data['pulse']
t = data['time']

################## ADAPTIVE FILTER PARAMETERS ##################################
L = 50
mu =  2e-4 #4.9e-4 #10**-5
sample_num = len(train)
delay = 5 #number of sample delays
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
for i in range(0,sample_num-L):
    x_n = x_delta[i:i+L]                      #section of the recieved signal
    output[i] = np.dot(x_n,w)           #section * filter weights
    e = x[i+L] - output[i]              #error for lms
    w = w + 2*mu*e*x_n                  #update filter weights
 
    if i == sample_num-L-1:             #store the last 
        optimal += w
 
filtered = np.concatenate([output, np.zeros(L)])#pad to match timeline
 
###############################################################################
'''
####################WORK WITH COMPLEX SIGNALS##################################
x = noisy     #recieved signal
d = train  #desired
w = m.zeros(L, 'complex')     #filter weights
w = np.matrix([w])
 
output = m.zeros((sample_num-L,1), 'complex')

  
for i in range(0,sample_num-L):
    x_n = x[i:i+L]             # slice of x relevant in this iteration
    x_n = np.matrix(x_n)
    output[i] = np.vdot(x_n,w.getH())        # multiply and add with filter weights
    e = d[i+L] - output[i]    # calculate error signal
    w = w + 2*mu*e*x_n         # update filter weights
    if i == sample_num-L-1:             #store the last 
        optimal = w

output = np.matrix(output).getH() #matrix form (can't be plotted
 
filtered = m.zeros(output.size, 'complex')
for i in range(output.size):
    filtered[i] = output[0, i]
     
  
filtered = np.concatenate([filtered, np.zeros(L)])
###############################################################################
'''
 
 
f_d = np.convolve(optimal, d)           #filtered desired
just_noise = x - d                      #just noise
f_n = np.convolve(just_noise, optimal)       #filtered noise

####calculating powers for snr calcs
Ps = np.zeros(len(f_d))
Pn = np.zeros(len(f_d))
for i in range(len(f_d)):
    Ps[i] = abs(f_d[i])**2
    Pn[i] = abs(f_n[i])**2
 
calcSNR = np.average(Ps)/np.average(Pn) #SNR in linear
calcSNR = 10 * np.log10(calcSNR)        #SNR in dB
################################################################################
 
################################### PLOTS ######################################
 
plt.subplot(2,1,1)
plt.plot(t, d, 'b')
plt.title('Transmit Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
     
plt.subplot(2,1,2)
plt.plot(t, x, 'g', label="x[n]")
plt.plot(t, filtered, 'r', label="y[n]")
plt.title('Desired Signal (x[n]) and Filtered signal (y[n])')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
 
#PLOT ORIGINALS
plt.subplot(2,1,1)
plt.plot(t, np.real(noisy))
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Real Signal + Noise: Fixed Frequency')
  
plt.subplot(2,1,2)
plt.plot(t, np.imag(noisy), 'r')
plt.title('Imaginary Signal + Noise: Fixed Frequency')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

FFT = np.fft.fft((np.real(noisy)))
FFT_db = np.zeros(len(FFT))
for i in range(len(FFT)):
    FFT_db[i] = 10*np.log10(FFT[i])
 
FFT_db = FFT_db[len(FFT_db)/2:]
plt.plot(FFT_db)
plt.title('FFT of Signal: Fixed Frequency')
plt.xlabel('Frequency')
plt.ylabel('Amplitude [dB]')
plt.show()
 
################################################################################