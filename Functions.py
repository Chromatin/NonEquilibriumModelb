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
    f=np.array(force)
    return 1 - 0.5*(np.sqrt(kBT/(f*p)))+(f/S)#returns Z_WLC as fraction of L

def hook(force,k=1,fmax=10):
    force=np.array(force)
    np.place(force,force>fmax,[10])        
    return force/k #returns Z_fiber as function of the number of bp in the fiber
    
def forcecalib(Pos,FMax=85): #Calculates Force from magnet position
    l1=1.4 #decay length 1 (mm)
    l2=0.8 #decay length 2 (mm)
    f0=0.01 #force-offset (pN)    
    return FMax*(0.7*np.exp(-Pos/l1)+0.3*np.exp(-Pos/l2))+f0

def findpeaks(y,n=15): #Finds y peaks at position x in xy graph
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
    x=np.array(x)
    a = (8*(np.pi-3)) / (3*np.pi*(4-np.pi))
    b = -x**2*(4/np.pi+a*x**2)/(1+a*x**2)
    return np.sign(x) * np.sqrt(1-np.exp(b))

def state2step(States):
    States=np.array(States)
    if States.size>1:
        return States[1:]-States[0:-1]
    else: return []

def ratio(Lmin,Lmax,x):
    FiberLength=Lmax-Lmin
    Ratio=((FiberLength-(x-Lmin))/(FiberLength) )   
    if Ratio <=0 :
        Ratio = 0
    if Ratio >=1:
        Ratio = 1
    return Ratio

def fjc(f, k_pN_nm = 0.1, b = None,  L_nm = 1, S_pN = 1e3):
    if b == None:
        b = 3 * kBT / (k_pN_nm * L_nm)
    x = f*b/kBT
    z = np.array([])
    for xi in x:
        z=np.append(z,L_nm*(sympy.coth(xi) -1/xi))
    z += L_nm*f/S_pN
    return z

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
    
def pdf(x,step,sigma):
    return 1-erfaprox((x+step)/sigma*np.sqrt(2))
    
def fit_pdf(y):
    y=np.array([y])
    y=np.sort(y)
    x=np.linspace(0,1,np.size(y))
    return curve_fit(pdf, y, x)
    
    