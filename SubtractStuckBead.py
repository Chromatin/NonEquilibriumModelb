# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:35:29 2018

@author: nhermans

Corrects .dat files from multiplexed magnetic tweezers, using stuck beads 
within the sample. Plots the substracted signal and the beads that have been
used to generate this reference. Always visually inspect this trace and judge 
take care the subtracted signal actually makes sense. After the reference 
subtraction, the used reference beads should be essentially straight lines.

"""

###############################################################################
#############   How to use:
#############   1) Put all the .dat files you want to correct in one folder
#############   2) Copy-Paste the foldername down below
#############   3) Rename the Subfolder for the corrected .dat files
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import csv
import os 
from scipy import signal
plt.close()
           

folder = r'G:\Klaas\Tweezers\Yeast Chromatin\Regensburg_18S\2018\2018_02_02_dUAF_Reg\FC3 15ul'
newpath = folder+r'\CorrectedDat'   
if not os.path.exists(newpath):
    os.makedirs(newpath)
    
###############################################################################
##########################   Functions   ######################################
###############################################################################

def read_dat(Filename):
    f = open(Filename, 'r')
    #get headers
    headers = f.readlines()[0]
    headers.rstrip()
    headers = headers.split('\t')
    f.close()  
    #get data
    data = genfromtxt(Filename, skip_header = 1)
    return data, headers
    
def subtract_reference(data, headers, Beads=3, MedianFilter=11, XY_correct = False):
    """Open .dat file from magnetic tweezers, averages the least moving beads and substracts them from the signal. 
    Output is a 2D array with all the data
    ***kwargs:
        Beads = number of beads to use for averaging, default = 3
        MedianFilter = LowPass filter for applied to averaged signal, default = 5. Needs to be an odd number
    """
    T = data[:,headers.index('Time (s)')]
    Z_all = data[:,headers.index('Z0'+' (um)')::4]
    X_all = data[:,headers.index('X0'+' (um)')::4]
    Y_all = data[:,headers.index('Y0'+' (um)')::4]
    Z_std =  np.std(X_all, axis=0) * np.std(Y_all, axis=0) * np.std(Z_all, axis=0)
    Z = Z_all[:,np.nanargmin(Z_std)]
    fit = np.polyfit(np.append(T[:100], T[len(T)-100:len(T)]),np.append(Z[:100], Z[len(Z)-100:len(Z)]),1)
    fit_fn = np.poly1d(fit)                              # fit_fn is a function which takes in x and returns an estimate for y  
#    plt.scatter(T,fit_fn(T), color = 'g')
    
    Z_DriftCorrected = np.subtract(Z_all, np.tile(fit_fn(T),[len(Z_all[0,:]),1]).T)
    Z_std =  np.std(X_all, axis=0) * np.std(Y_all, axis=0) * np.std(Z_all, axis=0)
    dZ = np.nanmax(Z_DriftCorrected,axis=0) - np.nanmin(Z_DriftCorrected,axis=0)
    Z_std = dZ * Z_std
    
    AveragedStuckBead = np.zeros(len(T))
    AveragedStuckBead_X, AveragedStuckBead_Y = AveragedStuckBead, AveragedStuckBead
    StuckBead=np.array([])
    mean=0
    ReferenceBeads = []
    
    for i in range(0,Beads):
        Low = np.nanargmin(Z_std)
        ReferenceBeads = np.append(ReferenceBeads,Low)
        StuckBead = Z_all[:,Low]
        mean += np.mean(StuckBead)
        StuckBead = np.subtract(StuckBead,np.mean(StuckBead))
        StuckBead = np.nan_to_num(StuckBead)
        AveragedStuckBead = np.sum([AveragedStuckBead,StuckBead/Beads], axis=0)
        AveragedStuckBead_X = np.sum([AveragedStuckBead_X,X_all[:,Low]/Beads], axis=0)
        AveragedStuckBead_Y = np.sum([AveragedStuckBead_Y,Y_all[:,Low]/Beads], axis=0)
        Z_std[Low] = np.nan
        
    mean = mean / Beads    
    AveragedStuckBead = signal.medfilt(AveragedStuckBead,MedianFilter)
    AveragedStuckBead_X = signal.medfilt(AveragedStuckBead_X,MedianFilter)
    AveragedStuckBead_Y = signal.medfilt(AveragedStuckBead_Y,MedianFilter)
    if XY_correct == False:
        for i,x in enumerate(Z_std):
            Position = headers.index('Z'+str(i)+' (um)')
            data[:,Position] = np.subtract(data[:,Position], AveragedStuckBead + mean )
    else:
        for i,x in enumerate(Z_std):
            Position = headers.index('Z'+str(i)+' (um)')
            data[:,Position] = np.subtract(data[:,Position], AveragedStuckBead + mean )
            data[:,Position-2] = np.subtract(data[:,Position-2], AveragedStuckBead_X )
            data[:,Position-1] = np.subtract(data[:,Position-1], AveragedStuckBead_Y )
    
#    for i in ReferenceBeads:
#        plt.scatter(T,data[:,headers.index('Z'+str(int(i))+' (um)')], alpha=0.5, label=str(i), lw=0) 
    return ReferenceBeads, Z_std, AveragedStuckBead, headers, data

###############################################################################
##########################     Script    ######################################
###############################################################################

filenames = os.listdir(folder)
os.chdir(folder)
    
Filenames = []                                                                  #All .dat files    
for filename in filenames:
    if filename[-4:] == '.dat':
        Filenames.append(filename)

for Filenum, DatFile in enumerate(Filenames):
    try: data, headers = read_dat(DatFile)
    except OSError: 
        print('>>>>>>>>>>>>File ', DatFile,' skipped>>>>>>>>>' ) 
        continue
    try: ReferenceBeads, Z_std, AveragedStuckBead, headers, data = subtract_reference(data, headers, Beads = 5, MedianFilter = 11)
    except ValueError:
        print('>>>>>>>>>>>>Value error in ', DatFile,', probably a calibration file missing Z data>>>>>>>>>' )
        continue
    
    plt.figure(Filenum)
    T = data[:,headers.index('Time (s)')]
    plt.scatter(T,AveragedStuckBead, color = 'b')
    plt.title(DatFile)
    plt.xlabel('time (s)')
    plt.ylabel('Z (um)')
    
    for i in ReferenceBeads:
        plt.scatter(T,data[:,headers.index('Z'+str(int(i))+' (um)')], alpha=0.5, label=str(i), lw=0)
        #plt.scatter(T,data_original[:,headers.index('Z'+str(int(i))+' (um)')], alpha=0.5, label=str(i), lw=0, color=plt.cm.cool(i))
    plt.legend(loc='best')
    plt.show()
    
    with open(newpath +'\\'+ DatFile, 'w') as outfile:          #writes new .dat file  
        writer = csv.writer(outfile, delimiter ='\t', lineterminator="\r") 
        headers[len(headers)-1] = "Amp a.u."
        data=np.vstack([np.array(headers), data])
        for row in data:
            writer.writerow(row)
