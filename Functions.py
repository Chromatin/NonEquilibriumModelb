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

def erfaprox(x):
    """Approximation of the error function"""
    x = np.array(x)
    a = (8*(np.pi-3)) / (3*np.pi*(4-np.pi))
    b = -x**2*(4/np.pi+a*x**2)/(1+a*x**2)
    return np.sign(x) * np.sqrt(1-np.exp(b))

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

def gaus(x,amp,x0,sigma):
    """1D Gaussian"""
    return amp*np.exp(-(x-x0)**2/(2*sigma**2))

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
      
