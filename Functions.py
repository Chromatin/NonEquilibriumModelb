# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:44:01 2018

@author: nhermans
"""
import numpy as np
from scipy import signal
from scipy import stats

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

def fjc(f, par): 
    """calculates a Freely Jointed Chain with a kungslength of b""" 
    #Function is independent on length of the DNA #L_nm = par['L_bp']*par['DNAds_nm']
    b = 3 * par['kBT_pN_nm'] / (par['k_pN_nm'])#*L_nm)
    x = f * b / par['kBT_pN_nm']
    # coth(x)= (exp(x) + exp(-x)) / (exp(x) - exp(x)) --> see Wikipedia
    exp_x = np.exp(x)
    z = (exp_x + 1 / exp_x) / (exp_x - 1 / exp_x) - 1 / x
    #z *= par['L_bp']*par['DNAds_nm']
    #z_df = (par['kBT_pN_nm'] / b) * (np.log(np.sinh(x)) - np.log(x))  #*L_nm #  + constant --> integrate over f (finish it
    #w = f * z - z_df
    return z

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
    Ratio = np.array((Par['LFiber_bp']-(x-Par['FiberStart_bp']))/(Par['LFiber_bp']))
    Ratio[Ratio<=0] = 0                                                         #removes values below 0, makes them 0
    Ratio[Ratio >=1] = 1                                                        #removes values above 1, makes them 1
    return np.abs(Ratio)

def localstiffness(F,Z,PossibleStates,Par,Fmax_Hook=10):
    """Calculates the probability landscape of the intermediate states. 
    F is the Force Data, 
    Z is the Extension Data (needs to have the same size as F)"""
    States = np.transpose(np.tile(PossibleStates,(len(F),1))) #Copies PossibleStates array into colomns of States with len(F) rows
    Ratio = ratio(PossibleStates, Par)
    Ratio = np.tile(Ratio,(len(F),1))
    Ratio = np.transpose(Ratio)
    dF = 0.01 #delta used to calculate the RC of the curve
    StateExtension = np.array(np.multiply(wlc(F, Par),(States*Par['DNAds_nm'])) + np.multiply(hook(F,Par['k_pN_nm'],Fmax_Hook),Ratio)*Par['ZFiber_nm'])
    StateExtension_dF = np.array(np.multiply(wlc(F+dF, Par),(States*Par['DNAds_nm'])) + np.multiply(hook(F+dF,Par['k_pN_nm'],Fmax_Hook),Ratio)*Par['ZFiber_nm'])
    DeltaZ = abs(np.subtract(StateExtension,Z))
    LocalStiffness = dF / np.subtract(StateExtension_dF,StateExtension)         #[pN/nm]            #*Par['kBT_pN_nm']
    return LocalStiffness, DeltaZ

#Including Hookian    
def probsum(F,Z,PossibleStates,Par,Fmax_Hook=10):
    """Calculates the probability landscape of the intermediate states. 
    F is the Force Data, 
    Z is the Extension Data (needs to have the same size as F)"""
    LocalStiffness, DeltaZ = localstiffness(F, Z, PossibleStates, Par, Fmax_Hook)
    sigma = np.sqrt(Par['kBT_pN_nm']/LocalStiffness)    
    NormalizedDeltaZ = np.divide(DeltaZ,sigma)    
    Pz = np.array((1-erfaprox(NormalizedDeltaZ)))
    ProbSum = np.sum(Pz, axis=1) 
    return ProbSum

#Including FJC
#def probsum(F,Z,PossibleStates,Par,Fmax_Hook=10):
#    """Calculates the probability landscape of the intermediate states using a combination of Freely jointed chain for the fiber, and Worm like chain for the DNA. 
#    F is the Force Data, 
#    Z is the Extension Data (needs to have the same size as F)
#    Stepsize is the precision -> how many possible states are generated. Typically 1 for each bp unwrapped"""
#    States = np.transpose(np.tile(PossibleStates,(len(F),1))) #Copies PossibleStates array into colomns of States with len(F) rows
#    Ratio = ratio(PossibleStates, Par)
#    Ratio = np.tile(Ratio,(len(F),1))
#    Ratio = np.transpose(Ratio)
#    dF = 0.01 #delta used to calculate the RC of the curve
#    StateExtension = np.array(np.multiply(wlc(F, Par),(States*Par['DNAds_nm'])) + np.multiply(fjc(F,Par),Ratio)*Par['DNAds_nm'])
#    StateExtension_dF = np.array(np.multiply(wlc(F+dF, Par),(States*Par['DNAds_nm'])) + np.multiply(fjc(F+dF,Par),Ratio)*Par['DNAds_nm'])
#    LocalStiffness = np.subtract(StateExtension_dF,StateExtension)*Par['kBT_pN_nm'] / dF 
#    DeltaZ = abs(np.subtract(StateExtension,Z))
#    Std = np.divide(DeltaZ,np.sqrt(LocalStiffness))
#    Pz = np.array(np.multiply((1-erfaprox(Std)),F))
#    ProbSum = np.sum(Pz, axis=1) 
#    return ProbSum

def gaus(x,amp,x0,sigma):
    """1D Gaussian"""
    return amp*np.exp(-(x-x0)**2/(2*sigma**2))

def removestates(StateMask, n=5):
    """Removes states with less than n data points, returns indexes of states to be removed"""
    RemoveStates = np.array([])    
    for i in np.arange(0,len(StateMask[0,:]),1):
        if sum(StateMask[:,i]) < n:
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

def find_states_prob(F_Selected, Z_Selected, Z, Force, Pars, MergeStates=False, P_Cutoff=0.1):
    """Finds states based on the probablitiy landscape"""     
    #Generate FE curves for possible states
    PossibleStates = np.arange(Pars['FiberStart_bp']-200, Pars['L_bp']+50,1)    #range to fit 
    ProbSum = probsum(F_Selected, Z_Selected, PossibleStates, Pars)             #Calculate probability landscape
    PeakInd, Peak = findpeaks(ProbSum, 25)                                      #Find Peaks    
    States = PossibleStates[PeakInd]                                            #Defines state for each peak

    AllStates = np.empty(shape=[len(Z), len(States)])                           #2d array of the states  
    AllStates_Selected = np.empty(shape=[len(Z_Selected), len(States)])                           #2d array of the states  \  
    for i, x in enumerate(States):
        Ratio = ratio(x,Pars)
        Fit = np.array(wlc(Force,Pars)*x*Pars['DNAds_nm'] + hook(Force,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        Fit_Selected = np.array(wlc(F_Selected,Pars)*x*Pars['DNAds_nm'] + hook(F_Selected,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        AllStates[:,i] = Fit        
        AllStates_Selected[:,i] = Fit_Selected        
    
    LocalStiffness, DeltaZ = localstiffness(F_Selected, Z_Selected, States, Pars)
            
    sigma = np.sqrt(Pars['kBT_pN_nm']/LocalStiffness)    
    std = np.sqrt(Pars['MeasurementERR (nm)']**2 + np.multiply(sigma,sigma))                         #sqrt([measuring error]^2 + [thermal fluctuations]^2) 
    Z_Score = z_score(Z_Selected, AllStates_Selected, std, States)    

    StateMask = np.abs(Z_Score) < 2.5
    PointsInState = np.sum(StateMask, axis=0)
    
#    #Remove states with 5 or less datapoints
    RemoveStates = removestates(StateMask)
    if len(RemoveStates)>0:
        States = np.delete(States, RemoveStates)
        Peak = np.delete(Peak, RemoveStates)
        PeakInd = np.delete(PeakInd, RemoveStates)
        StateMask = np.delete(StateMask, RemoveStates, axis=1)
        AllStates = np.delete(AllStates, RemoveStates, axis=1)
        AllStates_Selected = np.delete(AllStates_Selected, RemoveStates, axis=1)

    #Merging 2 states and checking whether is better or not
    for i in np.arange(0,len(States)-1): 
        MergedState = (AllStates_Selected[:,i]*PointsInState[i]+AllStates_Selected[:,i+1]*PointsInState[i+1])/(PointsInState[i]+PointsInState[i+1]) #weighted average over 2 neigbouring states
        
        LocalStiffness, DeltaZ = localstiffness(F_Selected, Z_Selected, MergeStates, Pars)
        print(np.shape(LocalStiffness))        
        sigma = np.sqrt(Pars['kBT_pN_nm']/LocalStiffness)    
        std = np.sqrt(Pars['MeasurementERR (nm)']**2 + np.multiply(sigma,sigma))                         #sqrt([measuring error]^2 + [thermal fluctuations]^2) 
        Z_Score = z_score(Z_Selected, MergedState, std, 1)
        print(np.shape(Z_Score))
        MergedStateMask = np.abs(Z_Score) < 2.5
        MergedSum = np.sum(MergedStateMask)
#        print("# Of point within 2.5 sigma in State1:State2:Merged =", PointsInState[i],":", PointsInState[i+1], ":", MergedSum)


    
#    T_test=np.array([])

#    for i, x in enumerate(States):
#        if i > 0:
#            Prob = stats.ttest_ind((StateMask == i) * Z_Selected, (StateMask == i - 1) * Z_Selected,equal_var=False)  # get two arrays for t_test
#            T_test = np.append(T_test, Prob[1])

#    while MergeStates == True:  # remove states untill all states are significantly different
#    
#        # Merges states that are most similar, and are above the p_cutoff minimal significance t-test value
#        HighP = np.argmax(T_test)
#        if T_test[HighP] > P_Cutoff:  # Merge the highest p-value states
#            DelState, MergedState = HighP, HighP+1
#            if sum((StateMask == HighP + 1) * 1) < sum((StateMask == HighP) * 1): 
#                DelState = HighP + 1
#                MergedState = HighP
#            #Recalculate th     
#            Prob = stats.ttest_ind((StateMask == MergedState ) * Z_Selected, (StateMask == HighP - 1) * Z_Selected,equal_var=False)  # get two arrays for t_test
#            T_test[HighP-1] = Prob[1]
#            Prob = stats.ttest_ind((StateMask == MergedState ) * Z_Selected, (StateMask == HighP - 1) * Z_Selected,equal_var=False)  # get two arrays for t_test
#            if len(T_test) > HighP+1:
#                T_test[HighP+1] = Prob[1]
#            T_test=np.delete(T_test,HighP)
#            States = np.delete(States, DelState)  # deletes the state in the state array
#            StateMask = StateMask - (StateMask == HighP + 1) * 1  # merges the states in the mask
#            Z_NewState = (StateMask == HighP) * Z_Selected  # Get all the data for this state to recalculate mean
#            MergeStates = True
#        else:
#            MergeStates = False  # Stop merging states
#                   
#        #calculate the number of L_unrwap for the new state
#        if MergeStates:
#            #find value for merged state with gaus fit / mean
#            StateProbSum = probsum(F_Selected[Z_NewState != 0],Z_NewState[Z_NewState != 0],PossibleStates,Pars)
#            States[HighP] = PossibleStates[np.argmax(StateProbSum)]  
            
    return PossibleStates, ProbSum, Peak, States, AllStates, StateMask


def z_score(Z_Selected, Z_States, std, States):
    """Calculate the z score of each value in the sample, relative to the a given mean and standard deviation.
    Parameters:	
            a : array_like
            An array like object containing the sample data.
            mean: float
            std : float
    """
    States = np.array([States])
    Z_Selected_New = (np.tile(Z_Selected,(len(States),1))).T               #Copies Z_Selected array into colomns of States with len(Z_States[0,:]) rows
    Z_States = np.tile(Z_States, (1, 1))    
    return np.divide(Z_Selected_New-Z_States, std.T)
    
def RuptureForces(Z_Selected, F_Selected, States, Pars, ax1):
    """Calculate and plot the rupture forces and jumps"""
    MedFilt = signal.medfilt(Z_Selected, 9)
#    MedFiltMask = attribute2state(F_Selected, MedFilt, States, Pars)               #For the Median Filter to which state it belongs
#    k = 0
#    for i, j in enumerate(MedFiltMask):
#        if j > k:
#            start = MedFilt[i-1]
#            stop = MedFilt[i]
#            ax1.hlines(F_Selected[i], start, stop, color='black', lw = 2)
#        k = j      
    ax1.plot(MedFilt, F_Selected, color='black')
