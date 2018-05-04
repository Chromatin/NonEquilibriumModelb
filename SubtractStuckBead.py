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

plt.close()

def read_dat(Filename, Av=5):
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
        #if Beadnumber == 0: Z_all = Z
        #else: Z_all = np.append(Z_all,Z, axis=0)
        Beadnumber+=1
    AveragedStuckBead = np.zeros(len(Z))
    StuckBead=np.array([])
    for i in range(0,Av):
        Low = np.argmin(Z_all)
        Position = headers.index('Z'+str(Low)+' (um)')
        StuckBead = data[:,Position]
        StuckBead = np.subtract(StuckBead,np.mean(StuckBead))
        AveragedStuckBead = np.sum([AveragedStuckBead,StuckBead], axis=0)
        Z_all[Low] = 1
        
    AveragedStuckBead = np.divide(AveragedStuckBead,Av) 
    for i,x in enumerate(Z_all):
        Position = headers.index('Z'+str(i)+' (um)')
        data[:,Position] = np.subtract(data[:,Position],AveragedStuckBead)
    
    return AveragedStuckBead, headers, data


DatFile = r'G:\Klaas\Tweezers\Tests\data_014.dat'

AveragedStuckBead, headers, data = read_dat(DatFile)
writer = csv.writer(open(DatFile+'.tmp', 'w'), delimiter ='\t')

writer.writerow(headers)
for row in data:
    writer.writerow(row)
