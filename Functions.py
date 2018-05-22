# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:44:01 2018

@author: nhermans
"""
import numpy as np
from scipy import signal

def wlc(force,Pars): #in nm/pN, as fraction of L
    """Calculates WLC in nm/pN, as a fraction the Contour Length.
    Returns Z_WLC as fraction of L """
    f = np.array(force)
    return 1 - 0.5*(np.sqrt(Pars['kBT_pN_nm']/(f*Pars['P_nm'])))+(f/Pars['S_pN'])

def hook(force, k=1, fmax=10):
    """Calculates Hookian in nm/pN
    Returns Z_fiber as function of the number of bp in the fiber"""
    f = np.array(force)
    np.place(f,f>fmax,[fmax])
    return f/k 

def exp(x):
    return np.exp(x)

def fjc(f, Pars): 
    """calculates a Freely Jointed Chain with a kungslength of 
    b = 3 KbT / k*L
    where L is the length of the fiber in nm, and k the stiffness in nm pN per nucleosome""" 
    
    b = 3 * Pars['kBT_pN_nm'] / (Pars['k_pN_nm']*Pars['ZFiber_nm'])
    x = f * b / Pars['kBT_pN_nm']
    # coth(x)= (exp(x) + exp(-x)) / (exp(x) - exp(x)) --> see Wikipedia
    z = (exp(x) + 1 / exp(x)) / (exp(x) - 1 / exp(x)) - 1 / x
    #z *= Pars['L_bp']*Pars['DNAds_nm']   #work /dG term not used atm
    #z_df = (Pars['kBT_pN_nm'] / b) * (np.log(np.sinh(x)) - np.log(x))  #*L_nm #  + constant --> integrate over f (finish it
    #w = f * z - z_df
    return z * (Pars['N_tot']-Pars['N4'])

def forcecalib(Pos,FMax=85): 
    """Calibration formula for 0.8mm gapsize magnet
    Calculates Force from magnet position"""
    l1 = 1.4 #decay length 1 (mm)
    l2 = 0.8 #decay length 2 (mm)
    f0 = 0.01 #force-offset (pN)    
    return FMax*(0.7*np.exp(-Pos/l1)+0.3*np.exp(-Pos/l2))+f0

def findpeaks(y,n=25):
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

def findpeaks_simple(y):
    """Finds y peaks at position x in xy graph"""
    y = np.array(y)
    y = conv(y, box_pts=25)
    peaks_index=np.array([])
    peaks_height=np.array([])
    for i,x in enumerate(y[1:-1]):
        if x > y[i] and  x > y[i+2]:
            peaks_index=np.append(peaks_index,i+1)
            peaks_height=np.append(peaks_height,x)
    return peaks_index.astype(int), peaks_height

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
    if Pars['LFiber_bp']<=0:
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
    StateExtension = np.array(np.multiply(wlc(F, Pars),(States*Pars['DNAds_nm'])) + np.multiply(fjc(F,Pars),Ratio))
    StateExtension_dF = np.array(np.multiply(wlc(F+dF, Pars),(States*Pars['DNAds_nm'])) + np.multiply(fjc(F+dF,Pars),Ratio))
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
    StateExtension = np.array(np.multiply(wlc(F, Pars),(States*Pars['DNAds_nm'])) + np.multiply(fjc(F,Pars),Ratio))
    StateExtension_dF = np.array(np.multiply(wlc(F+dF, Pars),(States*Pars['DNAds_nm'])) +  np.multiply(fjc(F+dF,Pars),Ratio))
    DeltaZ = abs(np.subtract(StateExtension,Z))
    LocalStiffness = dF / np.subtract(StateExtension_dF,StateExtension)         #[pN/nm]            #*Pars['kBT_pN_nm']    
    sigma = np.sqrt(Pars['kBT_pN_nm']/LocalStiffness)    
    NormalizedDeltaZ = np.divide(DeltaZ,sigma)    
    Pz = np.array((1-erfaprox(NormalizedDeltaZ)))
    ProbSum = np.sum(Pz, axis=1) 
    return ProbSum

def STD_hook(F, Z, PossibleStates, Pars, Fmax_Hook=10):
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
def probsum_hook(F, Z, PossibleStates, Pars, Fmax_Hook=10):
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

def find_states_prob(F_Selected, Z_Selected, F, Z, Pars, MergeStates=False, Z_Cutoff=2):
    """Finds states based on the probablitiy landscape and merges where necessary"""     
    #Generate FE curves for possible states
    start = Pars['FiberStart_bp']-200
    if start <= 0: start = 1 
    PossibleStates = np.arange(start, Pars['L_bp']+50,1)    #range to fit 
    ProbSum = probsum(F_Selected, Z_Selected, PossibleStates, Pars)             #Calculate probability landscape
    PeakInd, Peak = peakdetect(ProbSum, delta=1)                                #Find Peaks    
    States = PossibleStates[PeakInd]                                            #Defines state for each peak

    #2d array of the states: Containts coloms of states with each entry in a row the extension corresponding to the force in F(_Selected)  
    AllStates = np.empty(shape=[len(Z), len(States)])                           
    AllStates_Selected = np.empty(shape=[len(Z_Selected), len(States)])  
    for i, x in enumerate(States):
        Ratio = ratio(x,Pars)
        Fit = np.array(wlc(F,Pars)*x*Pars['DNAds_nm'] + hook(F,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        Fit_Selected = np.array(wlc(F_Selected,Pars)*x*Pars['DNAds_nm'] + hook(F_Selected,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        AllStates[:,i] = Fit        
        AllStates_Selected[:,i] = Fit_Selected        
    
    std = STD(F_Selected, Z_Selected, States, Pars)
    Z_Score = z_score(Z_Selected, AllStates_Selected, std, States)    
    
    StateMask = np.abs(Z_Score) < Z_Cutoff
    PointsPerState = np.sum(StateMask, axis=0)

    #Remove states with 'Minpoints' or less datapoints
    RemoveStates = removestates(StateMask, MinPoints=1)
    if len(RemoveStates)>0:
        States = np.delete(States, RemoveStates)
        Peak = np.delete(Peak, RemoveStates)
        PeakInd = np.delete(PeakInd, RemoveStates)
        StateMask = np.delete(StateMask, RemoveStates, axis=1)
        AllStates = np.delete(AllStates, RemoveStates, axis=1)
        AllStates_Selected = np.delete(AllStates_Selected, RemoveStates, axis=1)
        Z_Score = np.delete(Z_Score, RemoveStates, axis=1)
    
    PointsPerState = np.sum(StateMask, axis=0)

    #Merging 2 states and checking whether is better or not
    #Make copies of all crucial arrays, so the original maintains its values
    NewStates = np.copy(States)
    NewStateMask = np.copy(StateMask)
    NewAllStates = np.copy(AllStates)
    NewZ_Score = np.copy(Z_Score)
    
    N_Merged = 0                                                                #Keeps track of the number merges that have taken place
    for i in np.arange(0,len(States)-1): 
        i = i - N_Merged #Correct for the states that are removed

        MergedState = (NewStates[i]*PointsPerState[i]+NewStates[i+1]*PointsPerState[i+1])/(PointsPerState[i]+PointsPerState[i+1]) #New state that is weigheted average of two neighbouring states
        Ratio = ratio(MergedState,Pars)
        MergedStateArr = np.array(wlc(F,Pars)*MergedState*Pars['DNAds_nm'] + hook(F,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        MergedStateArr_Selected = np.array(wlc(F_Selected,Pars)*MergedState*Pars['DNAds_nm'] + hook(F_Selected,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        
        Std = STD(F_Selected, Z_Selected, MergedState, Pars)
        Z_Score_MergedState = z_score(Z_Selected, MergedStateArr_Selected, Std, 1).ravel()
        
        MergedStateMask = np.abs(Z_Score_MergedState) < Z_Cutoff
#        MergedStateMask = MergedStateMask.ravel() ###################Not necessary anymore?
        MergedSum = np.sum(MergedStateMask)
        
        #Fraction of overlapping points in datapoints between two initial states                      
        Overlap = np.sum(NewStateMask[:,i] * NewStateMask[:,i+1])/np.min([PointsPerState[i], PointsPerState[i+1]]) 
        
        #Fraction overlapping points in datapoints between the merged state and sum of two initial states 
        overlap = np.sum(np.any([NewStateMask[:,i],NewStateMask[:,i+1]], axis=0)*MergedStateMask*1)/np.max([np.sum(NewStateMask[:,i]),np.sum(NewStateMask[:,i+1])])
        
        #Merge Overlapping states when new state is better:
        if Overlap  > 0.5 and overlap > 0.8:       
            #Delete 1 of the two initial states            
            NewStates = np.delete(NewStates, i)
            PointsPerState = np.delete(PointsPerState, i)
            NewStateMask = np.delete(NewStateMask, i, axis=1)
            NewAllStates = np.delete(NewAllStates, i, axis=1)
            NewZ_Score = np.delete(NewZ_Score, i, axis=1)
            #Replace the other state by the new (merged) state
            NewStates[i] = MergedState
            PointsPerState[i] = MergedSum
            NewStateMask[:,i] = MergedStateMask
            NewAllStates[:,i] = MergedStateArr
            NewZ_Score[:,i] = Z_Score_MergedState
            N_Merged += 1
                          
    return PossibleStates, ProbSum, Peak, States, AllStates, StateMask, NewStates, NewAllStates, NewStateMask

def conv(y, box_pts=5):
    """Convolution of a signal y with a box of size box_pts with height 1/box_pts"""
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

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

def double_gauss(x, step=75, Sigma=15, a1=1, a2=1):
    return a1*(1+erfaprox((x-step)/(Sigma*np.sqrt(2))))+a2*(1+erfaprox((x-(step*2))/(Sigma*np.sqrt(2))))

def double_indep_gauss(x, step1=80, step2=160, Sigma=15, a1=1, a2=1):
    return a1*(1+erfaprox((x-step1)/(Sigma*np.sqrt(2))))+a2*(1+erfaprox((x-step2)))/(Sigma*np.sqrt(2))

def fit_2step_gauss(Steps, Step = 80, Amp1 = 30, Amp2 = 10, Sigma = 15):
    """Function to fit 25nm steps with a double gauss, as a PDF"""
    from scipy.optimize import curve_fit
    Steps = np.array(Steps)
    Steps = np.sort(Steps)
    PDF = np.arange(len(Steps))
    #popt, pcov = curve_fit(double_indep_gauss, Steps, PDF, p0=[Step, 2*Step, Sigma, Amp1, Amp2])
    popt, pcov = curve_fit(double_gauss, Steps, PDF, p0=[Step, Sigma, Amp1, Amp2])
    return popt

def attribute2state(F, Z, States, Pars, Fmax_Hook=10):
    """Calculates for each datapoint which state it most likely belongs too
    Return an array with indexes referring to the State array"""
    Ratio = ratio(States,Pars)
    WLC = wlc(F,Pars).reshape(len(wlc(F,Pars)),1)
    Hook = hook(F,Pars['k_pN_nm'],Fmax_Hook).reshape(len(hook(F,Pars['k_pN_nm'],Fmax_Hook)),1)
    ZState = np.array( np.multiply(WLC,(States*Pars['DNAds_nm'])) + np.multiply(Hook,(Ratio*Pars['ZFiber_nm'])) )
    ZminState = np.subtract(ZState,Z.reshape(len(Z),1)) 
    StateMask = np.argmin(abs(ZminState),1)       
    return StateMask 
   
def RuptureForces(F_Selected, Z_Selected, T_Selected, States, Pars, ax1, ax3):
    """
    Calculates rupture forces and and corresponding stepsizes in bp by applying
    a median filter over the data. Plots the median filtered data over the
    Force-Extension curve (ax1), and in the timetrace curve (ax3).
    Return--F_Rup_up: the forces for jumps to a higher state (tuple); 
            Step_up: stepsizes for jumps to a higher state (tuple);
            F_Rup_down: the forces for jumps to a lower state (tuple); 
            Step_down: stepsizes for jumps to a lower state (tuple)
    """

    dt = (T_Selected[-1]-T_Selected[0])/len(T_Selected)    
    
    Mask = attribute2state(F_Selected, Z_Selected, States, Pars)                #Tels to which state a datapoint belongs
    MedianFilt = signal.medfilt(Mask, 9)
    
    AllStates_Selected = np.empty(shape=[len(Z_Selected), len(States)])     
    for i, x in enumerate(States):
        Ratio = ratio(x,Pars)
        Fit_Selected = np.array(wlc(F_Selected,Pars)*x*Pars['DNAds_nm'] + hook(F_Selected,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        AllStates_Selected[:,i] = Fit_Selected        
 
    NonEqFit = []                                                               #Highlights the occupied state at a given time/force
    k = 0                                                                       #For the first loop
    F_Rup_up, Step_up, F_Rup_down, Step_down = [], [], [], []                   #Rupture forces and corresponding jumps
    TotalLifetime = np.zeros([len(States),])                                    #Total lifetime for each State in seconds
    for i, j in enumerate(MedianFilt):    
        j = int(j)
        TotalLifetime[int(j)] += 1        
        NonEqFit.append(AllStates_Selected[i,int(j)])
        if k > j:
            F_Rup_down.append(F_Selected[i])
            Step_down.append((AllStates_Selected[i,j+1]-AllStates_Selected[i,j])/Pars['DNAds_nm'])
        if k < j:
            F_Rup_up.append(F_Selected[i])
            Step_up.append((AllStates_Selected[i,j]-AllStates_Selected[i,j-1])/Pars['DNAds_nm'])
        k = j
    
    TotalLifetime *= dt

    ax1.plot(NonEqFit, F_Selected, color='black', lw=2)
    ax3.plot(T_Selected, NonEqFit, color='black', lw=2)

    return F_Rup_up, Step_up, F_Rup_down, Step_down

def BrowerToland(F_Selected, Z_Selected, T_Selected, States, Pars, ax1, ax3):
    """Returns a 2D-array with coloms 'ruptureforce', 'Number of nucleosomes left', 'dF/dt'"""

    dt = (T_Selected[-1]-T_Selected[0])/len(T_Selected)                         #Time interval
    
    Mask = Z_Selected > ( Pars['Fiber0_bp']  * Pars['DNAds_nm'] ) - 15  #Only select data from the start of the bead on the string state, - 1 Std                
    F_Selected = F_Selected[Mask]
    Z_Selected = Z_Selected[Mask]
    T_Selected = T_Selected[Mask]
        
    Mask = attribute2state(F_Selected, Z_Selected, States, Pars)                #Tels to which state a datapoint belongs
    MedianFilt = signal.medfilt(Mask, 9)
    
    AllStates_Selected = np.empty(shape=[len(Z_Selected), len(States)])         #2D Array containing extensions for each state
    for i, x in enumerate(States):
        Ratio = ratio(x,Pars)
        Fit_Selected = np.array(wlc(F_Selected,Pars)*x*Pars['DNAds_nm'] + hook(F_Selected,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        AllStates_Selected[:,i] = Fit_Selected        
 
    NonEqFit = []                                                               #Highlights the occupied state at a given time/force
    k = 10000                                                                   #For the first loop 
    F_Rup = np.empty((0,3)) #np.array([RuptureForce, N-nucl left, dF/dt])
    TotalLifetime = np.zeros([len(States),])
    for i, j in enumerate(MedianFilt):    
        j = int(j)
        TotalLifetime[int(j)] += 1        
        NonEqFit.append(AllStates_Selected[i,int(j)])
        DeltaZ = AllStates_Selected[i,int(j)]-AllStates_Selected[i,int(j-1)]
        if k < j and DeltaZ > 20 and DeltaZ < 30:                               #Only analyse 25 +- 5 nm steps
            N = round((wlc(F_Selected[i-1], Pars)*Pars['L_bp']-Z_Selected[i]/Pars['DNAds_nm'])/79) #Number of nucl left at i, rounded to the nearest int.
            dF_dt = (F_Selected[i]-F_Selected[i-1])/dt
            F_Rup = np.append(F_Rup, [[F_Selected[i-1], N, dF_dt]], axis=0)    
        k = j
    
    TotalLifetime *= dt
    
    ax1.plot(NonEqFit, F_Selected, color='blue', lw=2)
    ax3.plot(T_Selected, NonEqFit, color='blue', lw=2)
    return F_Rup

def BrowerToland_Stacks(F_Selected, Z_Selected, T_Selected, States, Pars, ax1, ax3):
    """Returns a 2D-array with coloms 'ruptureforce', 'Number of nucleosomes left', 'dF/dt'"""

    dt = (T_Selected[-1]-T_Selected[0])/len(T_Selected)                         #Time interval
    
    Mask = Z_Selected < ( Pars['Fiber0_bp']  * Pars['DNAds_nm'] ) + 15  #Only select data up to the start of the bead on the string state, + 1 Std                
    F_Selected = F_Selected[Mask]
    Z_Selected = Z_Selected[Mask]
    T_Selected = T_Selected[Mask]
        
    Mask = attribute2state(F_Selected, Z_Selected, States, Pars)                #Tels to which state a datapoint belongs
    MedianFilt = signal.medfilt(Mask, 9)
    
    AllStates_Selected = np.empty(shape=[len(Z_Selected), len(States)])         #2D Array containing extensions for each state
    for i, x in enumerate(States):
        Ratio = ratio(x,Pars)
        Fit_Selected = np.array(wlc(F_Selected,Pars)*x*Pars['DNAds_nm'] + hook(F_Selected,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        AllStates_Selected[:,i] = Fit_Selected        
 
    NonEqFit = []                                                               #Highlights the occupied state at a given time/force
    k = 10000                                                                   #For the first loop 
    BT = np.empty((0,3)) #np.array([RuptureForce, N-nucl left, dF/dt])
    TotalLifetime = np.zeros([len(States),])
    for i, j in enumerate(MedianFilt):    
        j = int(j)
        TotalLifetime[int(j)] += 1        
        NonEqFit.append(AllStates_Selected[i,int(j)])
        DeltaZ = AllStates_Selected[i,int(j)]-AllStates_Selected[i,int(j-1)]
        if k < j and DeltaZ > 0:                                               #Only analyse Steps larger than 0nm
            dF_dt = (F_Selected[i]-F_Selected[i-1])/dt
            BT = np.append(BT, [[F_Selected[i-1], j, dF_dt]], axis=0)           
        k = j
    
    TotalLifetime *= dt
    if len(BT[:,1]) > 0:
        BT[:,1] = np.abs(BT[:,1]-np.max(BT[:,1])) + 1                           #Tels how much states are left

    ax1.plot(NonEqFit, F_Selected, color='green', lw=2)
    ax3.plot(T_Selected, NonEqFit, color='green', lw=2)
    return BT


def dG_browertoland(ln_dFdt_N, RFs, Pars):
    """ 
    Linear fit of the BT plot (a + bx)
    Calculates d (distance to transition) and K_d0 (energy of transition)
    Calculates errors of the fit and the propagated error in the K_d and d
    For error propagation: http://teacher.nsrl.rochester.edu/phy_labs/AppendixB/AppendixB.html
    """
    
    Fit = np.polyfit(ln_dFdt_N, RFs, 1, full = True)
    a = Fit[0][0]
    b = Fit[0][1]
    d = Pars['kBT_pN_nm']/a
    K_d0 = np.exp(-b/a)/a
    
    def d_err(a, d_a, Pars):
        return Pars['kBT_pN_nm']/a*(d_a/a) 
        
    def k_D0_err(a, d_a, b, d_b, Pars):
        d_ab = b/a*((d_b/b)**2+(d_a/a)**2)**(1/2)
        d_e_ab = np.exp(-b/a)*d_ab
        return 1/a*np.exp(-b/a)*((d_e_ab/np.exp(-a/b))**2+(d_a/a)**2)**(1/2) 
    
    a_err = Fit[3][0]
    b_err = Fit[3][1]

    D_err = d_err(a, a_err, Pars)
    K_d0_err = k_D0_err(a, a_err, b, b_err, Pars)
    
    K0 = 5e9
    Delta_G = -np.log(K_d0/K0) #in k_BT    
    Delta_G_err = K_d0_err/(K_d0)
    
    return a, a_err, b, b_err, d, D_err, K_d0, K_d0_err, Delta_G, Delta_G_err

def peakdetect(y_axis, lookahead = 10, delta=1.5):
    """
    Converted from/based on a MATLAB script at: 
    http://billauer.co.il/peakdet.html
    
    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively
    
    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200) 
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function
    
    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*tab)
    """
    min_peaks = []
    max_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
       
    # store data length for later use
    length = len(y_axis)
    x_axis = range(len(y_axis))
   
    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], 
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]
    
    
    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass
    
    if max_peaks == []: #Sometimes max_peaks == [], but why ?
        return [], []
    
    max_peaks=np.array(max_peaks)
        
    return max_peaks[:,0].astype(int), max_peaks[:,1]


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
