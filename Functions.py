# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:44:01 2018

@author: nhermans
"""
import numpy as np
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
    """Calculates distances between states""" #Not used atm in Cluster part
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
    Ratio = (Par['LFiber_bp']-(x-Par['FiberStart_bp']))/(Par['LFiber_bp']) 
    Ratio = np.array(Ratio)
    Ratiomin = Ratio<=0
    Ratio[Ratiomin] = 0         #removes values below 0, makes them 0
    RatioPlus = Ratio >=1
    Ratio[RatioPlus] = 1 #removes values above 1, makes them 1
    return np.abs(Ratio)

def probsum(F,Z,PossibleStates,Par,Fmax_Hook=10):
    """Calculates the probability landscape of the intermediate states. 
    F is the Force Data, 
    Z is the Extension Data (needs to have the same size as F)
    Stepsize is the precision -> how many possible states are generated. Typically 1 for each bp unwrapped"""
    States = np.transpose(np.tile(PossibleStates,(len(F),1))) #Copies PossibleStates array into colomns of States with len(F) rows
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

def removestates(StateMask, n=5):
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
     
def find_states_prob(F_Selected, Z_Selected, Pars, MergeStates=True, P_Cutoff=0.1):
    """Finds states based on the probablitiy landscape"""     
    from scipy import stats
    #Generate FE curves for possible states
    PossibleStates = np.arange(Pars['FiberStart_bp']-200, Pars['L_bp']+50,1)    #range to fit 
    ProbSum = probsum(F_Selected, Z_Selected, PossibleStates, Pars)        #Calculate probability landscape
    PeakInd, Peak = findpeaks(ProbSum, 25)                                 #Find Peaks    
    States = PossibleStates[PeakInd]                                            #Defines state for each peak

    #Calculate for each datapoint which state it most likely belongs too 
    StateMask = attribute2state(F_Selected,Z_Selected,States,Pars)
    
    #Remove states with 5 or less datapoints
    RemoveStates = removestates(StateMask)
    if len(RemoveStates)>0:
        States = np.delete(States, RemoveStates)
        StateMask = attribute2state(F_Selected, Z_Selected, States, Pars)

    T_test=np.array([])
    
    for i, x in enumerate(States):
        if i > 0:
            Prob = stats.ttest_ind((StateMask == i) * Z_Selected, (StateMask == i - 1) * Z_Selected,equal_var=False)  # get two arrays for t_test
            T_test = np.append(T_test, Prob[1])

    while MergeStates == True:  # remove states untill all states are significantly different
    
        # Merges states that are most similar, and are above the p_cutoff minimal significance t-test value
        HighP = np.argmax(T_test)
        if T_test[HighP] > P_Cutoff:  # Merge the highest p-value states
            DelState, MergedState = HighP, HighP+1
            if sum((StateMask == HighP + 1) * 1) < sum((StateMask == HighP) * 1): 
                DelState = HighP + 1
                MergedState = HighP
            #Recalculate th     
            Prob = stats.ttest_ind((StateMask == MergedState ) * Z_Selected, (StateMask == HighP - 1) * Z_Selected,equal_var=False)  # get two arrays for t_test
            T_test[HighP-1] = Prob[1]
            Prob = stats.ttest_ind((StateMask == MergedState ) * Z_Selected, (StateMask == HighP - 1) * Z_Selected,equal_var=False)  # get two arrays for t_test
            if len(T_test) > HighP+1:
                T_test[HighP+1] = Prob[1]
            T_test=np.delete(T_test,HighP)
            States = np.delete(States, DelState)  # deletes the state in the state array
            StateMask = StateMask - (StateMask == HighP + 1) * 1  # merges the states in the mask
            Z_NewState = (StateMask == HighP) * Z_Selected  # Get all the data for this state to recalculate mean
            MergeStates = True
        else:
            MergeStates = False  # Stop merging states
                   
        #calculate the number of L_unrwap for the new state
        if MergeStates:
            #find value for merged state with gaus fit / mean
            StateProbSum = probsum(F_Selected[Z_NewState != 0],Z_NewState[Z_NewState != 0],PossibleStates,Pars)
            States[HighP] = PossibleStates[np.argmax(StateProbSum)]  
            
        return States
    
def STD(F,Z,PossibleStates,Par,Fmax_Hook=10):
    """Calculates the probability landscape of the intermediate states. 
    F is the Force Data, 
    Z is the Extension Data (needs to have the same size as F)
    Stepsize is the precision -> how many possible states are generated. Typically 1 for each bp unwrapped"""
    States = np.transpose(np.tile(PossibleStates,(len(F),1))) #Copies PossibleStates array into colomns of States with len(F) rows
    Ratio = ratio(PossibleStates, Par)
    Ratio = np.tile(Ratio,(len(F),1))
    Ratio = np.transpose(Ratio)
    dF = 0.01 #delta used to calculate the RC of the curve
    StateExtension = np.array(np.multiply(wlc(F, Par),(States*Par['DNAds_nm'])) + np.multiply(hook(F,Par['k_pN_nm'],Fmax_Hook),Ratio)*Par['ZFiber_nm'])
    StateExtension_dF = np.array(np.multiply(wlc(F+dF, Par),(States*Par['DNAds_nm'])) + np.multiply(hook(F+dF,Par['k_pN_nm'],Fmax_Hook),Ratio)*Par['ZFiber_nm'])
    LocalStiffness = np.subtract(StateExtension_dF,StateExtension)*Par['kBT_pN_nm'] / dF 
    DeltaZ = abs(np.subtract(StateExtension,Z))
    Std = np.divide(DeltaZ,np.sqrt(LocalStiffness))
    return Std
    
def Conv(y, box_pts):
    """Convolution of a signal y with a box of size box_pts with indeces 1/box_pts"""
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
