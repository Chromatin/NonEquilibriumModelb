# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:44:01 2018

@author: nhermans
"""
from __future__ import absolute_import

import numpy as np
from scipy import signal
from scipy import stats

def wlc(force,Pars): #in nm/pN, as fraction of L
    """Calculates WLC in nm/pN, as a fraction the Contour Length.
    Returns Z_WLC as fraction of L """
    f = np.array(force)
    return 1 - 0.5*(np.sqrt(Pars['kBT_pN_nm']/(f*Pars['P_nm'])))+(f/Pars['S_pN'])

def hook(force,k=1,fmax=10):
    """Calculates Hookian in nm/pN
    Returns Z_fiber as function of the number of bp in the fiber"""
    f = np.array(force)
    np.place(f,f>fmax,[fmax])
    return f/k 

def exp(x):
    return np.exp(x)

def fjc(f, Pars): 
    """calculates a Freely Jointed Chain with a kungslength of b""" 
    #Function is independent on length of the DNA #L_nm = Pars['L_bp']*Pars['DNAds_nm']
    b = 3 * Pars['kBT_pN_nm'] / (Pars['k_pN_nm'])#*L_nm)
    x = f * b / Pars['kBT_pN_nm']
    # coth(x)= (exp(x) + exp(-x)) / (exp(x) - exp(x)) --> see Wikipedia
    z = (exp(x) + 1 / exp(x)) / (exp(x) - 1 / exp(x)) - 1 / x
    #z *= Pars['L_bp']*Pars['DNAds_nm']
    #z_df = (Pars['kBT_pN_nm'] / b) * (np.log(np.sinh(x)) - np.log(x))  #*L_nm #  + constant --> integrate over f (finish it
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

def gaus(x, amp, x0, sigma):
    """1D Gaussian"""
    return amp*np.exp(-(x-x0)**2/(2*sigma**2))

def state2step(States):
    """Calculates distances between states"""    
    States = np.array(States)
    if States.size>1:
        return States[1:]-States[0:-1]
    else: return []

def ratio(x, Pars):
    """Calculates the number of Nuclesomes in the fiber, where 1 = All nucs in fiber and 0 is no Nucs in fiber. 
    Lmin = Unwrapped bp with fiber fully folded
    Lmax = Countour length of the DNA in the beads on a string conformation, where the remaining nucleosomes are still attached
    Imputs can be arrays"""
    if Pars['LFiber_bp']<0:
        return x*0
    Ratio = np.array((Pars['LFiber_bp']-(x-Pars['FiberStart_bp']))/(Pars['LFiber_bp']))
    Ratio[Ratio<=0] = 0                                                         #removes values below 0, makes them 0
    Ratio[Ratio>=1] = 1                                                         #removes values above 1, makes them 1
    return np.abs(Ratio)

def STD(F, Z, PossibleStates, Pars, Fmax_Hook=10):
    """Calculates the probability landscape of the intermediate states. 
    F is the Force Data, 
    Z is the Extension Data (needs to have the same size as F)"""
    States = np.transpose(np.tile(PossibleStates,(len(F),1))) #Copies PossibleStates array into colomns of States with len(F) rows
    Ratio = ratio(PossibleStates, Pars)
    Ratio = np.tile(Ratio,(len(F),1))
    Ratio = np.transpose(Ratio)
    dF = 0.01 #delta used to calculate the RC of the curve
    StateExtension = np.array(np.multiply(wlc(F, Pars),(States*Pars['DNAds_nm'])) + np.multiply(hook(F,Pars['k_pN_nm'],Fmax_Hook),Ratio)*Pars['ZFiber_nm'])
    StateExtension_dF = np.array(np.multiply(wlc(F+dF, Pars),(States*Pars['DNAds_nm'])) + np.multiply(hook(F+dF,Pars['k_pN_nm'],Fmax_Hook),Ratio)*Pars['ZFiber_nm'])
    LocalStiffness = dF / np.subtract(StateExtension_dF,StateExtension)         #[pN/nm]            #*Pars['kBT_pN_nm']    
    sigma = np.sqrt(Pars['kBT_pN_nm']/LocalStiffness)    
    std = np.sqrt(Pars['MeasurementERR (nm)']**2 + np.square(sigma))    #sqrt([measuring error]^2 + [thermal fluctuations]^2)  
    return std

#Including Hookian    
def probsum(F, Z, PossibleStates, Pars, Fmax_Hook=10):
    """Calculates the probability landscape of the intermediate states. 
    F is the Force Data, 
    Z is the Extension Data (needs to have the same size as F)"""
    States = np.transpose(np.tile(PossibleStates,(len(F),1))) #Copies PossibleStates array into colomns of States with len(F) rows
    Ratio = ratio(PossibleStates, Pars)
    Ratio = np.tile(Ratio,(len(F),1))
    Ratio = np.transpose(Ratio)
    dF = 0.01 #delta used to calculate the RC of the curve
    StateExtension = np.array(np.multiply(wlc(F, Pars),(States*Pars['DNAds_nm'])) + np.multiply(hook(F,Pars['k_pN_nm'],Fmax_Hook),Ratio)*Pars['ZFiber_nm'])
    StateExtension_dF = np.array(np.multiply(wlc(F+dF, Pars),(States*Pars['DNAds_nm'])) + np.multiply(hook(F+dF,Pars['k_pN_nm'],Fmax_Hook),Ratio)*Pars['ZFiber_nm'])
    DeltaZ = abs(np.subtract(StateExtension,Z))
    LocalStiffness = dF / np.subtract(StateExtension_dF,StateExtension)         #[pN/nm]            #*Pars['kBT_pN_nm']    
    sigma = np.sqrt(Pars['kBT_pN_nm']/LocalStiffness)    
    NormalizedDeltaZ = np.divide(DeltaZ,sigma)    
    Pz = np.array((1-erfaprox(NormalizedDeltaZ)))
    ProbSum = np.sum(Pz, axis=1) 
    return ProbSum

def find_states_prob(F_Selected, Z_Selected, F, Z, Pars, MergeStates=False, P_Cutoff=0.1):
    """Finds states based on the probablitiy landscape"""     
    #Generate FE curves for possible states
    PossibleStates = np.arange(Pars['FiberStart_bp']-200, Pars['L_bp']+50,1)    #range to fit 
    ProbSum = probsum(F_Selected, Z_Selected, PossibleStates, Pars)             #Calculate probability landscape
    PeakInd, Peak = findpeaks(ProbSum, 25)                                      #Find Peaks    
    States = PossibleStates[PeakInd]                                            #Defines state for each peak

    AllStates = np.empty(shape=[len(Z), len(States)])                           #2d array of the states  
    AllStates_Selected = np.empty(shape=[len(Z_Selected), len(States)])  
    for i, x in enumerate(States):
        Ratio = ratio(x,Pars)
        Fit = np.array(wlc(F,Pars)*x*Pars['DNAds_nm'] + hook(F,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        Fit_Selected = np.array(wlc(F_Selected,Pars)*x*Pars['DNAds_nm'] + hook(F_Selected,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        AllStates[:,i] = Fit        
        AllStates_Selected[:,i] = Fit_Selected        
    
    std = STD(F_Selected, Z_Selected, States, Pars)
    z_Score = z_score(Z_Selected, AllStates_Selected, std, States)    
    
    StateMask = np.abs(z_Score) < 2.5
    PointsPerState = np.sum(StateMask, axis=0)
#    #Remove states with 5 or less datapoints
    RemoveStates = removestates(StateMask, MinPoints=3)
    if len(RemoveStates)>0:
        States = np.delete(States, RemoveStates)
        Peak = np.delete(Peak, RemoveStates)
        PeakInd = np.delete(PeakInd, RemoveStates)
        StateMask = np.delete(StateMask, RemoveStates, axis=1)
        AllStates = np.delete(AllStates, RemoveStates, axis=1)
        AllStates_Selected = np.delete(AllStates_Selected, RemoveStates, axis=1)
    
    PointsPerState = np.sum(StateMask, axis=0)

    #Merging 2 states and checking whether is better or not
    NewStates = np.copy(States)
    NewStateMask = np.copy(StateMask)
    NewAllStates = np.copy(AllStates)
    k = 0
    for i in np.arange(0,len(States)-1): 
        i = i - k
        MergedState = (NewStates[i]*PointsPerState[i]+NewStates[i+1]*PointsPerState[i+1])/(PointsPerState[i]+PointsPerState[i+1])
        Ratio = ratio(MergedState,Pars)
        MergedStateArr = np.array(wlc(F_Selected,Pars)*MergedState*Pars['DNAds_nm'] + hook(F_Selected,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        MergedStateAllArr = np.array(wlc(F,Pars)*MergedState*Pars['DNAds_nm'] + hook(F,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        
        Std = STD(F_Selected, Z_Selected, MergedState, Pars)
        Z_Score = z_score(Z_Selected, MergedStateArr, Std, 1)
        
        MergedStateMask = np.abs(Z_Score) < 2.5
        MergedStateMask = MergedStateMask.ravel()
        MergedSum = np.sum(MergedStateMask)
#        print("# Of point within 2.5 sigma in State", i, ":State", i+1, ":Merged =", PointsPerState[i],":", PointsPerState[i+1], ":", MergedSum)
        
        Diff = []
        Diff.append(NewStateMask[:,i] * NewStateMask[:,i+1])
        Diff.append(MergedStateMask * NewStateMask[:,i])
        Diff.append(MergedStateMask * NewStateMask[:,i+1])
        
        Overlap = []
        Overlap.append(len(Diff[0][Diff[0]==True])/np.min([PointsPerState[i], PointsPerState[i+1]]))
        Overlap.append(len(Diff[1][Diff[1]==True])/np.min([MergedSum, PointsPerState[i]]))
        Overlap.append(len(Diff[2][Diff[2]==True])/np.min([MergedSum, PointsPerState[i+1]]))

#        print(Overlap, PointsPerState[i], PointsPerState[i+1], MergedSum)
        

        if Overlap[0] > 0.6 and np.min(Overlap[1:]) > 0.5: #What criterium should be here?!
            NewStates = np.delete(NewStates, i)
            PointsPerState = np.delete(PointsPerState, i)
            NewStateMask = np.delete(NewStateMask, i, axis=1)
            NewAllStates = np.delete(NewAllStates, i, axis=1)
            NewStates[i] = MergedState
            PointsPerState[i] = MergedSum
            NewStateMask[:,i] = MergedStateMask
            NewAllStates[:,i] = MergedStateAllArr
            k += 1
                
    
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
            
    return PossibleStates, ProbSum, Peak, States, AllStates, StateMask, NewStates, NewStateMask, NewAllStates

def removestates(StateMask, MinPoints=5):
    """Removes states with less than n data points, returns indexes of states to be removed"""
    RemoveStates = np.array([])    
    for i in np.arange(0,len(StateMask[0,:]),1):
        if sum(StateMask[:,i]) < MinPoints:
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

def z_score(Z_Selected, Z_States, std, States):
    """Calculate the z score of each value in the sample, relative to the a given mean and standard deviation.
    Parameters:	
            a : array_like
            An array like object containing the sample data.
            mean: float
            std : float
    """
    if type(States) == np.ndarray: #while merging states, Z_States is only 1 state, this fixes dimensions
        Z_Selected_New = (np.tile(Z_Selected,(len(States),1))).T               #Copies Z_Selected array into colomns of States with len(Z_States[0,:]) rows    
    else:
        Z_Selected_New = np.reshape(Z_Selected, (len(Z_Selected),1))
        Z_States = np.reshape(Z_States, (len(Z_States),1))
    return np.divide(Z_Selected_New-Z_States, std.T)


def ChiSquared(f, e):
    """e is the expected value and f is the observed frequency, and summed over all possibilities """    
    return np.sum(np.devide(np.square(f-e), e))
    
   
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

def T_Test(a, b, var_a, var_b):
    N = len(a)    
    s = np.sqrt((var_a + var_b)/2)
#    s
    
    ## Calculate the t-statistics
    t = (np.mean(a) - np.mean(b))/(s*np.sqrt(2/N))
    
    
    
    ## Compare with the critical t-value
    #Degrees of freedom
    df = 2*N - 2
    
    #p-value after comparison with the t 
    p = 1 - stats.t.cdf(t,df=df)
    
    
#    print("t = " + str(t))
#    print("p = " + str(2*p))
    #Note that we multiply the p value by 2 because its a twp tail t-test
    ### You can see that after comparing the t statistic with the critical t value (computed internally) we get a good p value of 0.0005 and thus we reject the null hypothesis and thus it proves that the mean of the two distributions are different and statistically significant.
    
    
    ## Cross Checking with the internal scipy function
    t2, p2 = stats.ttest_ind(a,b)
#    print("t = " + str(t2))
#    print("p = " + str(2*p2))
    return p, p2
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
