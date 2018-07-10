# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:52:17 2018

@author: nhermans
"""
#from lmfit import Parameters

import numpy as np
from scipy import signal

def Define_Handles(Select=True, Pull=True, DelBreaks=True, MinForce=2, MaxForce=True, MinZ=0, MaxZ=False, Onepull=True, MedFilt=False):
    """If analysis has to be done on only part of the data, these options can be used"""
    Handles = {}
    Handles['Select'] = Select
    Handles['Pulling'] = Pull
    Handles['DelBreaks'] = DelBreaks
    Handles['MinForce'] = MinForce
    Handles['MaxForce'] = MaxForce
    Handles['MinZ'] = MinZ
    Handles['MaxZ'] = MaxZ
    Handles['MedFilt'] = MedFilt
    Handles['Onepull'] = Onepull
    return Handles

def read_data(Filename):
    """Open .dat/.fit files from magnetic tweezers"""
    f = open(Filename, 'r')
    #get headers
    headers = f.readlines()[0]
    headers = headers.split('\t')
    #get data
    data = np.genfromtxt(Filename, skip_header = 1)
    F = data[:,headers.index('F (pN)')]
    Z = data[:,headers.index('z (um)')]*1000  #Z in nm
    T = data[:,headers.index('t (s)')]
    Z_Selected = data[:,headers.index('selected z (um)')]*1000

    return F, Z, T, Z_Selected

def read_log(Filename):
    """Open the corresponding .log files from magnetic tweezers. Returns False if the file is not found"""
    try: 
        f = open(Filename, 'r')
    except FileNotFoundError: 
        print(Filename, '========> No valid logfile found')
        return False   
    lines = f.readlines()
    f.close()
    return lines

def log_pars(LogFile):
    """Reads in parameters from the logfile generate by the labview fitting program, returns a {dict} with 'key'= paramvalue"""
    par = {}
    par['L_bp'] = float(find_param(LogFile, 'L DNA (bp)'))
    par['P_nm'] = float(find_param(LogFile, 'p DNA (nm)'))
    par['S_pN'] = float(find_param(LogFile, 'S DNA (pN)'))
    par['degeneracy'] = 0
    par['z0_nm'] = 2
    par['k_pN_nm'] = float(find_param(LogFile, 'k folded (pN/nm)'))
    par['N_tot'] = float(find_param(LogFile, 'N nuc'))
    par['N4'] = float(find_param(LogFile, 'N unfolded [F0]'))
    par['NRL_bp'] = float(find_param(LogFile, 'NRL (bp)'))
    par['ZFiber_nm'] = float(find_param(LogFile, 'l folded (nm)'))
    par['G1_kT'] = 3
    par['G2_kT'] = 4
    par['DNAds_nm'] = 0.34 # rise per basepair (nm)
    par['kBT_pN_nm'] = 4.2 #pn/nm 
    par['Innerwrap_bp'] = 79 #number of basepairs in the inner turn wrap
    par['Fiber0_bp'] = par['L_bp']-(par['N_tot']*par['Innerwrap_bp'])  #Transition between fiber and beats on a string
    par['LFiber_bp'] = (par['N_tot']-par['N4'])*(par['NRL_bp']-par['Innerwrap_bp'])  #total number of bp in the fiber
    par['FiberStart_bp']  = par['Fiber0_bp']-par['LFiber_bp']
    par['MeasurementERR (nm)'] = 5                                                #tracking inaccuracy in nm
    return par

def find_param(Logfile, Param):
    """Find a parameter in the .log file"""
    for lines in Logfile:
        P =lines.split(' = ')
        if P[0]==Param:
            return P[1].strip('\n')
    print("<<<<<<<<<<", Param, "not found >>>>>>>>>>")
    return

def default_pars():
    """Default fitting parameters, returns a {dict} with 'key'= paramvalue"""
    par = {}
    par['L_bp']= 3040
    par['P_nm'] = 50
    par['S_pN'] = 1000
    par['degeneracy'] = 0
    par['z0_nm'] = 0
    par['N_tot'] = 0
    par['N4'] = 0
    par['NRL_bp'] = 167
    par['k_pN_nm'] = 1
    par['G1_kT'] = 3
    par['G2_kT'] = 4
    par['DNAds_nm'] = 0.34 # rise per basepair (nm)
    par['kBT_pN_nm'] = 4.2 #pn/nm 
    par['Innerwrap_bp'] = 79 #number of basepairs in the inner turn wrap
    par['Fiber0_bp']  = par['L_bp']-(par['N_tot']*par['Innerwrap_bp'])  #Transition between fiber and beats on a string
    par['LFiber_bp'] = (par['N_tot']-par['N4'])*(par['NRL_bp']-par['Innerwrap_bp'])  #total number of bp in the fiber
    par['FiberStart_bp'] = par['Fiber0_bp']-par['LFiber_bp'] #DNA handles
    par['MeasurementERR (nm)'] = 5     #tracking inaccuracy in nm
    return par

def handle_data(F, Z, T, Z_Selected, Handles, Pars=default_pars(), Window=5):
    """Can be used to remove data that disrupts proper fitting
    Please read 'handles' to see the options"""
    
    if Handles['Select']:                                                       #If only the selected column is use do this
        F_Selected = np.delete(F, np.argwhere(np.isnan(Z_Selected)))
        T_Selected = np.delete(T, np.argwhere(np.isnan(Z_Selected)))
        Z_Selected = np.delete(Z, np.argwhere(np.isnan(Z_Selected))) 
        if len(Z_Selected) == 0: 
            print('==> Nothing Selected!')
            return [], [], []
        else:
            F_Selected, Z_Selected, T_Selected = minforce(F_Selected, Z_Selected, T_Selected , Handles['MinForce'])
            return F_Selected, Z_Selected, T_Selected
    else:
        F_Selected = F
        Z_Selected = Z
        T_Selected = T
    
    if Handles['DelBreaks']: F_Selected ,Z_Selected, T_Selected = breaks(F_Selected, Z_Selected, T_Selected, Jump = 1500)
    if Handles['Pulling']: F_Selected, Z_Selected, T_Selected = removerelease(F_Selected, Z_Selected, T_Selected )
    if Handles['MinForce'] > 0: F_Selected, Z_Selected, T_Selected = minforce(F_Selected, Z_Selected, T_Selected , Handles['MinForce'])
    if Handles['MaxZ']:                                                         #Remove all datapoints after max extension
        Handles['MaxZ'] = (Pars['L_bp']+100)*Pars['DNAds_nm']
        F_Selected, Z_Selected, T_Selected = maxextention(F_Selected, Z_Selected, T_Selected , Handles['MaxZ']) #remove data above Z=1.1*LC
    if Handles['Onepull']: F_Selected, Z_Selected, T_Selected = onepull(F_Selected, Z_Selected, T_Selected, 10)
    if Handles['MedFilt']: Z_Selected = signal.medfilt(Z_Selected, Window)
    return F_Selected, Z_Selected, T_Selected

def breaks(F, Z, T, Jump=1000):
    """Removes the data after a jump in z, presumably indicating the bead broke lose"""
    LowPass = signal.medfilt(Z,3)
    extra = 0
    for i,x in enumerate(LowPass[1:]):
        change = abs(x - LowPass[i]) + extra
        if change > Jump :
            F = F[:i]
            Z = Z[:i] 
            T = T[:i] 
            break
        if abs(x - LowPass[i]) > 100:
            extra = change
        else: extra = 0
    return F, Z, T

def removerelease(F, Z, T):
    """Removes the release curve from the selected data"""
    F_diff = np.diff(F)
    F_diff = np.insert(F_diff,0,0)
    F = F[F_diff>=0]
    Z = Z[F_diff>=0]
    T = T[F_diff>=0]
    return F, Z, T

def maxforce(F, Z, T,  Max_Force=10):
    """Removes the data above Max force given"""
    T = T[F<Max_Force]
    Z = Z[F<Max_Force]
    F = F[F<Max_Force]
    return F, Z, T

def minforce(F, Z,  T, Min_Force=2):
    """Removes the data below minimum force given"""
    Z = Z[F>Min_Force]
    T = T[F>Min_Force]
    F = F[F>Min_Force]
    return F, Z, T

def maxextention(F, Z, T, Max_Extension): 
    """Removes the data above maximum extension given"""
    Mask = Z < Max_Extension
    Z = Z[Mask]
    F = F[Mask]
    T = T[Mask]
    return F, Z ,T

def onepull(F, Z, T, Jump=10):
    """Selects only the last pulling curve"""
    T_Jump = np.diff(T)
    mask = T_Jump > Jump
    ind = np.where(mask)[0]
    if len(ind)>0:    
        F = F[ind[-1]+1:]
        Z = Z[ind[-1]+1:]
        T = T[ind[-1]+1:]
    return F, Z, T
    
"""
#This function is not used atm
            
def write_data(Filename,Headers,Data):
    f = open(Filename, 'a')
#    import json
#    json.dump(str(Data),f)
    Headers='\t'.join(map(str,Headers))+'\n'
    f.write(Headers)
    Data='\t'.join(map(str,Data))+'\n'
    f.write(Data)
    f.close()
    return "resultsfile generated"
"""
