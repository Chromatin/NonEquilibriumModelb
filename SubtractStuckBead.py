# -*- coding: utf-8 -*-
"""
Created on Thu May 03 09:53:13 2018

@author: nhermans
"""

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
    #get data
    data = genfromtxt(Filename, skip_header = 1)
    f.close()
    Z_all = np.array([])
    Beadnumber = 0
    bead = True
    Z = np.array([])
    
    while bead == True:    
        try:
            Z = data[:,headers.index('Z'+str(Beadnumber)+' (um)')]
            Z_all = np.append(Z_all, np.std(Z))
        except:
            bead = False
            print('done at bead', Beadnumber)
            break
        Beadnumber+=1
    
    AveragedStuckBead = np.zeros(len(Z))
    StuckBead=np.array([])
    mean=0
    ReferenceBeads = []
    
    for i in range(0,Av):
        Low = np.nanargmin(Z_all) 
        ReferenceBeads = np.append(ReferenceBeads,Low)
        Position = headers.index('Z'+str(Low)+' (um)')
        StuckBead = data[:,Position]
        mean += np.mean(StuckBead)
        StuckBead = np.subtract(StuckBead,np.mean(StuckBead))
        StuckBead = np.nan_to_num(StuckBead)
        AveragedStuckBead = np.sum([AveragedStuckBead,StuckBead], axis=0)
        Z_all[Low] = 1
        
    mean = mean / Av    
    #AveragedStuckBead = signal.medfilt(np.divide(AveragedStuckBead,Av) - mean,5)
    
    for i,x in enumerate(Z_all):
        Position = headers.index('Z'+str(i)+' (um)')
        data[:,Position] = np.subtract(data[:,Position],AveragedStuckBead + mean)
    
    T = data[:,headers.index('Time (s)')]
    plt.scatter(T,AveragedStuckBead, color = 'b')
    for i in ReferenceBeads:
        plt.scatter(T,data[:,headers.index('Z'+str(int(i))+' (um)')])
    plt.show()
    
    return Z_all, AveragedStuckBead, headers, data

folder = r'G:\Klaas\Tweezers\Tests'
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
