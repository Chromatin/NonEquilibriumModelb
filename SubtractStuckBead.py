# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:35:29 2018

@author: nhermans
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import csv
import os 
from scipy import signal
plt.close()

def read_dat(Filename, Av=3):
    """Open .dat/.fit files from magnetic tweezers"""
    f = open(Filename, 'r')
    #get headers
    headers = f.readlines()[0]
    headers = headers.split('\t')
    f.close()  
    #get data
    data = genfromtxt(Filename, skip_header = 1)
    T = data[:,headers.index('Time (s)')]
    Z_all = data[:,headers.index('Z0'+' (um)')::4]
    Z_std =  np.std(Z_all, axis=0)
    Z = Z_all[:,np.nanargmin(Z_std)]
    fit = np.polyfit(np.append(T[:100], T[len(T)-100:len(T)]),np.append(Z[:100], Z[len(Z)-100:len(Z)]),1)
    fit_fn = np.poly1d(fit) 
    # fit_fn is now a function which takes in x and returns an estimate for y  
    plt.scatter(T,fit_fn(T), color = 'g')
    
    Z_std = np.std(np.subtract(Z_all, np.tile(fit_fn(T),[len(Z_all[0,:]),1]).T),axis=0)
    
    AveragedStuckBead = np.zeros(len(Z))
    StuckBead=np.array([])
    mean=0
    ReferenceBeads = []
    
    for i in range(0,Av):
        Low = np.nanargmin(Z_std)
        ReferenceBeads = np.append(ReferenceBeads,Low)
        StuckBead = Z_all[:,Low]
        mean += np.mean(StuckBead)
        StuckBead = np.subtract(StuckBead,np.mean(StuckBead))
        StuckBead = np.nan_to_num(StuckBead)
        AveragedStuckBead = np.sum([AveragedStuckBead,StuckBead/Av], axis=0)
        Z_std[Low] = 100
        
    mean = mean / Av    
    AveragedStuckBead = signal.medfilt(AveragedStuckBead,5)
   
    for i in ReferenceBeads:
        plt.scatter(T,data[:,headers.index('Z'+str(int(i))+' (um)')], alpha=0.5, label=str(i), lw=0, c=np.random.rand(3,1)) 
            
    for i,x in enumerate(Z_std):
        Position = headers.index('Z'+str(i)+' (um)')
        data[:,Position] = np.subtract(data[:,Position],AveragedStuckBead+mean)    
    
    plt.scatter(T,AveragedStuckBead, color = 'b')
    
    for i in ReferenceBeads:
        plt.scatter(T,data[:,headers.index('Z'+str(int(i))+' (um)')], alpha=0.5)
    plt.legend(loc='best')
    plt.show()
       
    return Z_std, AveragedStuckBead, headers, data

folder = r'C:\Users\lion\Desktop\test'
newpath = folder+r'\CorrectedDat'   

if not os.path.exists(newpath):
    os.makedirs(newpath)
filenames = os.listdir(folder)
os.chdir(folder)
    
Filenames = []                                                                  #All .fit files    
for filename in filenames:
    if filename[-4:] == '.dat':
        Filenames.append(filename)

for Filenum, DatFile in enumerate(Filenames):
    
    Z_all, AveragedStuckBead, headers, data = read_dat(DatFile)
    with open(newpath +'\\'+ DatFile, 'w') as outfile:    
        writer = csv.writer(outfile, delimiter ='\t') 
        writer.writerow(headers)
        for row in data:
            writer.writerow(row)
