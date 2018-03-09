# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:44:01 2018

@author: nhermans
"""
import numpy as np
from scipy.optimize import curve_fit
#import sys

def wlc(force,par): #in nm/pN, as fraction of L
    """Calculates WLC in nm/pN, as a fraction the Contour Length.
    Returns Z_WLC as fraction of L """
    f = np.array(force)
    return 1 - 0.5*(np.sqrt(par['kBT_pN_nm']/(f*par['P_nm'])))+(f/par['S_pN'])

def hook(force,k=1,fmax=10):
    """Calculates Hookian in nm/pN
    Returns Z_fiber as function of the number of bp in the fiber"""
    f = np.array(force)
    np.place(f,f>fmax,[fmax])        
    return f/k 
    
def forcecalib(Pos,FMax=85): 
    """Calibration formula for 0.8mm gapsize magnet
    Calculates Force from magnet position"""
    l1 = 1.4 #decay length 1 (mm)
    l2 = 0.8 #decay length 2 (mm)
    f0 = 0.01 #force-offset (pN)    
    return FMax*(0.7*np.exp(-Pos/l1)+0.3*np.exp(-Pos/l2))+f0

def findpeaks(y,n=10):
    """Peakfinder writen with Thomas Brouwer
    Finds y peaks at position x in xy graph"""
    y = np.array(y)
    Yy = np.append(y[:-1],y[::-1])
    yYy = np.append(y[::-1][:-1],Yy)
    from scipy.signal import argrelextrema
    maxInd = argrelextrema(yYy, np.greater,order=n)
    r = np.array(yYy)[maxInd] 
    a = maxInd[0]
    #discard all peaks for negative dimers
    peaks_index=[]
    peaks_height=[]
    for n,i in enumerate(a):
        i=1+i-len(y)
        if i >= 0 and i <= len(y):
            peaks_height.append(r[n])
            peaks_index.append(i)
    return peaks_index, peaks_height

def erfaprox(x):
    """Approximation of the error function"""
    x = np.array(x)
    a = (8*(np.pi-3)) / (3*np.pi*(4-np.pi))
    b = -x**2*(4/np.pi+a*x**2)/(1+a*x**2)
    return np.sign(x) * np.sqrt(1-np.exp(b))

def state2step(States):
    """Calculates distances between states"""
    States = np.array(States)
    if States.size>1:
        return States[1:]-States[0:-1]
    else: return []

def ratio(x, Par):
    """Calculates the number of Nuclesomes in the fiber, where 1 = All nucs in fiber and 0 is no Nucs in fiber. 
    Lmin = Unwrapped bp with fiber fully folded
    Lmax = Countour length of the DNA in the beads on a string conformation, where the remaining nucleosomes are still attached
    Imputs can be arrays"""
    if Par['LFiber_bp']<0:
        return x*0
    Ratio = (Par['LFiber_bp']-(x-Par['Fiber0_bp']))/(Par['LFiber_bp'])
    Ratiomin = Ratio>=0
    Ratio *= Ratiomin          #removes values below 0, makes them 0
    Ratiomin = Ratio >=1
    Ratio = Ratio * (Ratio<=1) #removes values above 1, makes them 1
    return Ratiomin + Ratio

def probsum(F,Z,PossibleStates,Par,Fmax_Hook=10):
    """Calculates the probability landscape of the intermediate states. 
    F is the Force Data, 
    Z is the Extension Data (needs to have the same size as F)
    Stepsize is the precision -> how many possible states are generated. Typically 1 for each bp unwrapped"""
    States = np.tile(PossibleStates,(len(F),1))
    States = np.transpose(States)
    Ratio = ratio(PossibleStates, Par)
    Ratio = np.tile(Ratio,(len(F),1))
    Ratio = np.transpose(Ratio)
    dF = 0.01 #delta used to calculate the RC of the curve
    StateExtension = np.array(np.multiply(wlc(F, Par),(States*Par['DNAds_nm'])) + np.multiply(hook(F,Par['k_pN_nm'],Fmax_Hook),Ratio)*Par['ZFiber_nm'])
    StateExtension_dF = np.array(np.multiply(wlc(F+dF, Par),(States*Par['DNAds_nm'])) + np.multiply(hook(F+dF,Par['k_pN_nm'],Fmax_Hook),Ratio)*Par['ZFiber_nm'])
    LocalStiffness = np.subtract(StateExtension_dF,StateExtension)*Par['kBT_pN_nm'] / dF 
    DeltaZ = abs(np.subtract(StateExtension,Z))
    Std = np.divide(DeltaZ,np.sqrt(LocalStiffness))
    Pz = np.array(np.multiply((1-erfaprox(Std)),F))
    ProbSum = np.sum(Pz, axis=1) 
    return ProbSum

def gaus(x,amp,x0,sigma):
    """1D Gaussian"""
    return amp*np.exp(-(x-x0)**2/(2*sigma**2))

def removestates(StateMask, n=2):
    """Removes states with less than n data points, returns indexes of states to be removed"""
    RemoveStates = np.array([])
    for i in np.arange(0,np.amax(StateMask),1):
        if sum(StateMask == i) < n:
            RemoveStates = np.append(RemoveStates,i)
    return RemoveStates

def mergestates(States,MergeStates):         
    """Merges states as specied in the second array. If two consequtive states are to be merged, only one is removed.
    Returns a new State array"""
    old = 0
    for i,x in enumerate(MergeStates):
        if x-old != 1: 
            States = np.delete(States, x)
            old = x
    return States    
        
def attribute2state(F,Z,States,Pars,Fmax_Hook=10):
    """Calculates for each datapoint which state it most likely belongs too
    Return an array with indexes referring to the State array"""
    if len(States) <1:
        print('No States were found')
        return False
        #sys.exit('No States were found')
    Ratio = ratio(States,Pars)
    WLC = wlc(F,Pars).reshape(len(wlc(F,Pars)),1)
    Hook = hook(F,Pars['k_pN_nm'],Fmax_Hook).reshape(len(hook(F,Pars['k_pN_nm'],Fmax_Hook)),1)
    ZState = np.array( np.multiply(WLC,(States*Pars['DNAds_nm'])) + np.multiply(Hook,(Ratio*Pars['ZFiber_nm'])) )
    ZminState = np.subtract(ZState,Z.reshape(len(Z),1)) 
    StateMask = np.argmin(abs(ZminState),1)        
    return StateMask    
    
def fjc(f, par): 
    """calculates a Freely Jointed Chain with a kungslength of b""" 
    L_nm = par['L_bp']*par['DNAds_nm']
    b = 3 * par['kBT_pN_nm'] / (par['k_pN_nm']*L_nm)
    x = f * b / par['kBT_pN_nm']
    # coth(x)= (exp(x) + exp(-x)) / (exp(x) - exp(x)) --> see Wikipedia
    exp_x = np.exp(x)
    z = (exp_x + 1 / exp_x) / (exp_x - 1 / exp_x) - 1 / x
    z *= par['L_bp']*par['DNAds_nm']
    #z_df = (par['kBT_pN_nm'] / b) * (np.log(np.sinh(x)) - np.log(x))  #*L_nm #  + constant --> integrate over f (finish it
    #w = f * z - z_df
    return z

def pdf(x,step=79,sigma=15):
    """calculates the probability distribution function for a mean of size step""" 
    return 1-erfaprox((x+step)/sigma*np.sqrt(2))
    
def fit_pdf(y):
    y = np.array([y])
    y = np.sort(y)
    x = np.linspace(0,1,np.size(y))
    #popt = curve_fit(lambda f, p: Fit_Pss(f,p),Fit_F,Fit_Z,p0=0.6)
    return curve_fit(lambda x, step: pdf(x), y, x)
      
