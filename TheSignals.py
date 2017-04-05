import numpy as np
import math as math
import matplotlib.pyplot as plt
import pylab as m


#signal design
def papersignals(signaltype):
    if signaltype == 'fixed':
        f1 = 500e6 #freq of pulse
        fs = 1280e6 #sampling freq
        tau = 1e-6 #pulse length
        n = np.arange(0, tau, 1/fs) #timeline for pulse
        A1 = 1 #amplitude
        PRI = 10e-6 #PRI
        tmax = 50e-6 #max time for train
        t = np.arange(0,tmax, 1/fs) #timeline for train
        t = t[:len(t)-1] #corrected for erroneous additional one sample
        psi = 0 #phaseshift 
        temp = np.arange(0, PRI, 1/fs) #timeline for one interval
        temp = temp[:len(temp)-1] #corrected for erroneous additional one sample 
        pulse = A1 * np.exp((1j * 2 * np.pi * f1 * n) + psi) #pulse
        pad = np.zeros(len(temp)-len(pulse)) #zero padding 
        train = np.concatenate([pulse, pad, pulse, pad, pulse, pad, pulse, pad, pulse, pad])  #train creation
        train = train[:len(t)] #corrected train
     
    elif signaltype == 'chirp':
        f0 = 500e6 #init freq
        f1 = 530e6 #final freq
        fs = 1280e6 #samping frequency
        tau = 1e-6; #pulse length
        B = (f1-f0) #bandwidth
        k = B/tau
        n = np.arange(0, tau, 1/fs) #timeline for pulse
        A1 = 1 #amplitude
        PRI = 10e-6 #PRI
        tmax = 50e-6 #max time for train
        t = np.arange(0,tmax, 1/fs) #timeline for train
        t = t[:len(t)-1] #corrected for erroneous additional one sample
        temp = np.arange(0, PRI, 1/fs) #timeline for one interval
        temp = temp[:len(temp)-1] #corrected for erroneous additional one sample
        pulse = A1 * np.exp(1j *2 * np.pi * (f0 * n + (k/2.0) * (n**2))) #pulse
        pad = np.zeros(len(temp)-len(pulse)) #zero padding
         
        train = np.concatenate([pulse, pad, pulse, pad, pulse, pad, pulse, pad, pulse, pad]) #train creation
        train = train[:len(t)] #corrected train

     
    elif signaltype == 'bar':
        
        f0 = 500e6 #init freq
        fs = 1280e6 #samping frequency
        tau = 1e-6 #pulse length
        n = np.arange(0, tau, 1/fs) #timeline for pulse
        PRI = 10e-6 #PRI
        tmax = 50e-6 #max time for train
        t = np.arange(0,tmax, 1/fs) #timeline for train
        t = t[:len(t)-1] #corrected for erroneous additional one sample
        temp = np.arange(0, PRI, 1/fs) #timeline for one interval
        temp = temp[:len(temp)-1] #corrected for erroneous additional one sample
        y = np.exp(1j *2 * np.pi * f0 * n) #pulse
        pulse = m.zeros(len(n), 'complex')
          
        def makeneg(arrayy):
            woah = m.zeros(len(arrayy), 'complex')
            for j in range(len(arrayy)):
                woah[j] += -1*arrayy[j]
              
            return woah
          
        T = np.ceil(len(pulse)/7)
        pulse[0:3*T] = y[0:3*T]
        pulse[3*T+1:5*T] = makeneg(y[3*T+1:5*T])
        pulse[5*T+1:6*T] = y[5*T+1:6*T]
        pulse[6*T+1:] = makeneg(y[6*T+1:])
        
        pad = np.zeros(len(temp)-len(pulse)) #zero padding
         
        train = np.concatenate([pulse, pad, pulse, pad, pulse, pad, pulse, pad, pulse, pad])  #train creation
        train = train[:len(t)] #corrected train
        
    pack = {'pulse': pulse, 'train': train, 'time': t}
    
    return pack

        
                  
                
                