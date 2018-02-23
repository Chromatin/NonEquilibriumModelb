# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:44:01 2018

@author: nhermans
"""
import numpy as np
import sympy
import scipy
from scipy.optimize import curve_fit

kBT = 4.2 #pn/nm 
#worm-like chain
def wlc(force,p=50,S=1000): #in nm/pN, as fraction of L
    """Calculates WLC in nm/pN, as a fraction the Contour Length """
    f=np.array(force)
    return 1 - 0.5*(np.sqrt(kBT/(f*p)))+(f/S)#returns Z_WLC as fraction of L

def hook(force,k=1,fmax=10):
    """Calculates Hookian in nm/pN"""
    force=np.array(force)
    np.place(force,force>fmax,[fmax])        
    return force/k #returns Z_fiber as function of the number of bp in the fiber
    
def forcecalib(Pos,FMax=85): #Calculates Force from magnet position
    """Calibration formula for 0.8mm gapsize magnet"""
    l1=1.4 #decay length 1 (mm)
    l2=0.8 #decay length 2 (mm)
    f0=0.01 #force-offset (pN)    
    return FMax*(0.7*np.exp(-Pos/l1)+0.3*np.exp(-Pos/l2))+f0

def findpeaks(y,n=15): #Finds y peaks at position x in xy graph
    """Peakfinder writen with Thomas Brouwer"""
    y=np.array(y)
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
    x=np.array(x)
    a = (8*(np.pi-3)) / (3*np.pi*(4-np.pi))
    b = -x**2*(4/np.pi+a*x**2)/(1+a*x**2)
    return np.sign(x) * np.sqrt(1-np.exp(b))

def state2step(States):
    """Calculates distances between states"""
    States=np.array(States)
    if States.size>1:
        return States[1:]-States[0:-1]
    else: return []

def ratio(Lmin,Lmax,x):
    """Calculates the number of Nuclesomes in the fiber, where 1 = All nucs in fiber and 0 is no Nucs in fiber. 
    Imputs can be arrays"""
    FiberLength=Lmax-Lmin
    Ratio=((FiberLength-(x-Lmin))/(FiberLength) )   
    ratiomin = Ratio >=0
    Ratio*=ratiomin
    ratiomin = Ratio >=1
    Ratio = Ratio * (Ratio <=1)
    Eindratio = ratiomin + Ratio
#    if Ratio <=0 :
#        Ratio = 0
#    if Ratio >=1:
#        Ratio = 1
    return Eindratio

def minforce(tested_array,array2,test):
    Curingtest=np.array([])
    for i,x in enumerate(tested_array):
        if x < test:
            Curingtest=np.append(Curingtest,i)
    tested_array=np.delete(tested_array, Curingtest)
    array2=np.delete(array2,Curingtest)
    return tested_array,array2

def breaks(ForceSelected,Z_Selected, test=500):
    test=Z_Selected[0]
    for i,x in enumerate(Z_Selected[1:]):
        if abs(x - test) > 500 :
            ForceSelected=ForceSelected[:i]
            Z_Selected=Z_Selected[:i] 
            break
        test=x
    return ForceSelected, Z_Selected 

def removerelease(ForceSelected,Z_Selected):
    test=0
    Pullingtest=np.array([])
    for i,x in enumerate(ForceSelected):
        if x < test:
            Pullingtest=np.append(Pullingtest,i)
        test=x
    ForceSelected=np.delete(ForceSelected, Pullingtest)
    Z_Selected=np.delete(Z_Selected,Pullingtest)
    return ForceSelected, Z_Selected 

def probsum(F,Z,LFiber_min,LFiber_max,DNALength,p=50, S=1000, Z_fiber=10, k=1, DNAds=0.34, Fmax_Hook=10, Stepsize=1,dF=0.1):
    """Calculates the probability landscape of the intermediate states. 
    F is the Force Data, 
    Z is the Extension Data (needs to have the same size as F)
    Stepsize is the size of individual steps used for """
    
    PossibleStates = np.arange(LFiber_min-200,DNALength+50,Stepsize) #range to fit 
    States=np.tile(PossibleStates,(len(F),1))
    States=np.transpose(States)
    dF=0.1 #Used to calculate local stiffness
    ProbSum=np.array([])
    Ratio=ratio(LFiber_min,LFiber_max,PossibleStates)
    Ratio=np.tile(Ratio,(len(F),1))
    Ratio=np.transpose(Ratio)
    
    StateExtension=np.array(np.multiply(wlc(F,p,S),States)*DNAds + np.multiply(hook(F,k,Fmax_Hook),Ratio)*Z_fiber)
    StateExtension_dF=np.array(np.multiply(wlc(F+dF,p,S),States)*DNAds + np.multiply(hook(F+dF,k,Fmax_Hook),Ratio)*Z_fiber)
    LocalStiffness = np.subtract(StateExtension_dF,StateExtension)*(kBT) / dF # fix the units of KBT (pN nm -> pN um)
    DeltaZ=abs(np.subtract(StateExtension,Z))
    std=np.divide(DeltaZ,np.sqrt(LocalStiffness))
    Pz=np.array(np.multiply((1-erfaprox(std)),np.sqrt(F)))
    ProbSum=np.sum(Pz, axis=1) 
    return ProbSum

#These functions do not work yet    
def fjcold(f, k_pN_nm = 0.1, b = None,  L_nm = 1, S_pN = 1e3):
    if b == None:
        b = 3 * kBT / (k_pN_nm * L_nm)
    x = f*b/kBT
    z = np.array([])
    for xi in x:
        z=np.append(z,L_nm*(sympy.coth(xi) -1/xi))
    z += L_nm*f/S_pN
    return z

def fjc(f, par =None,  Lmax = 20):
    p = par.valuesdict()
    b = 3 * kBT / (p['k_pN_nm']*Lmax)
    x = f*b/kBT
    exp_x = np.exp(x)
    z = (exp_x +1/exp_x)/(exp_x - 1/exp_x) -1/x
    z *= Lmax
    #w = (exp_x - 1/exp_x)/2*x
    return np.asarray(z) #np.asarray(w)

def pdf(x,step,sigma):
    return 1-erfaprox((x+step)/sigma*np.sqrt(2))
    
def fit_pdf(y):
    y=np.array([y])
    y=np.sort(y)
    x=np.linspace(0,1,np.size(y))
    #popt = curve_fit(lambda f, p: Fit_Pss(f,p),Fit_F,Fit_Z,p0=0.6)
    return curve_fit(lambda x, step: pdf(x), y, x)
      