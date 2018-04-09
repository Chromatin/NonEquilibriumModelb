# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:52:17 2018

@author: nhermans
"""
#from lmfit import Parameters

import numpy as np
from scipy import signal

def Define_Handles(Select=True, Pull=True, DelBreaks=True, MinForce=2, MinZ=0, MaxZ=False, MedFilt=False):
    """If analysis has to be done on only part of the data, these options can be used"""
    Handles = {}
    Handles['Select'] = Select
    Handles['Pulling'] = Pull
    Handles['DelBreaks'] = DelBreaks
    Handles['MinForce'] = MinForce
    Handles['MinZ'] = MinZ
    Handles['MaxZ'] = MaxZ
    Handles['MedFilt'] = MedFilt 
    return Handles

def read_data(Filename):
    """Open .dat/.fit files from magnetic tweezers"""
    f = open(Filename, 'r')
    #get headers
    headers = f.readlines()[0]
    headers = headers.split('\t')
    #get data
    f.seek(0) #seek to beginning of the file
    data = f.readlines()[1:]
    f.close()
    F = np.array([])
    Z = np.array([])
    T = np.array([])
    Z_Selected = np.array([])
    for idx,item in enumerate(data):                                            #Get all the data from the fitfile
        F = np.append(F,float(item.split()[headers.index('F (pN)')]))
        Z = np.append(Z,float(item.split()[headers.index('z (um)')])*1000)   
        T = np.append(T,float(item.split()[headers.index('t (s)')]))
        Z_Selected = np.append(Z_Selected,float(item.split()[headers.index('selected z (um)')])*1000)
    return F, Z, T, Z_Selected

def read_log(Filename):
    """Open the corresponding .log files from magnetic tweezers"""
    f = open(Filename, 'r')
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
    par['MeasurementERR (nm)'] = 5
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
    par['MeasurementERR (nm)'] = 5   
    return par

def handle_data(F, Z, T, Z_Selected, Handles, Pars=default_pars(), Window=5):
    """Reads in parameters from the logfile generate by the labview fitting program"""
    if Handles['Select']:                                                       #If only the selected column is use do this
        F_Selected = np.delete(F, np.argwhere(np.isnan(Z_Selected)))
        T_Selected = np.delete(T, np.argwhere(np.isnan(Z_Selected)))
        Z_Selected = np.delete(Z, np.argwhere(np.isnan(Z_Selected))) 
        if len(Z_Selected)==0: 
            print('==> Nothing Selected!')
            return [], [], []
        else:
            F_Selected, Z_Selected, T_Selected = minforce(F_Selected, Z_Selected, T_Selected , 1)
            return F_Selected, Z_Selected, T_Selected
    else:
        F_Selected = F
        Z_Selected = Z
        T_Selected = T
    
    if Handles['DelBreaks']: F_Selected ,Z_Selected, T_Selected = breaks(F_Selected, Z_Selected, T_Selected, 1000)
    if Handles['Pulling']: F_Selected, Z_Selected, T_Selected = removerelease(F_Selected, Z_Selected, T_Selected )
    if Handles['MinForce'] > 0: F_Selected, Z_Selected, T_Selected = minforce(F_Selected, Z_Selected, T_Selected , Handles['MinForce'])
    if Handles['MaxZ']:                                                         #Remove all datapoints after max extension
        Handles['MaxZ'] = (Pars['L_bp']+100)*Pars['DNAds_nm']
#        print((Pars['L_bp']+100)*Pars['DNAds_nm'], - Pars['L_bp']*Pars['DNAds_nm']*1.1) #what is this!?
        Z_Selected, F_Selected, T_Selected = minforce(Z_Selected, F_Selected, T_Selected , - Pars['L_bp']*Pars['DNAds_nm']*1.1) #remove data above Z=1.1*LC
        #Here 'maxextension()' should be uses, but what is 'Max_extension' ?
    if Handles['MedFilt']: Z_Selected = signal.medfilt(Z_Selected, Window)
    return F_Selected, Z_Selected, T_Selected

def breaks(F, Z, T, test=500):
    """Removes the data after a jump in z, presumably indicating the bead broke lose"""
    test = Z[0]
    for i,x in enumerate(Z[1:]):
        if abs(x - test) > 500 :
            F = F[:i]
            Z = Z[:i] 
            T = Z[:i] 
            break
        test = x
    return F, Z, T

def removerelease(F, Z, T):
    """Removes the release curve from the selected data"""
    test = 0
    Pullingtest = np.array([])
    for i,x in enumerate(F):
        if x < test:
            Pullingtest = np.append(Pullingtest,i)
        test = x
    F = np.delete(F, Pullingtest)
    Z = np.delete(Z, Pullingtest)
    T = np.delete(T, Pullingtest)
    return F, Z, T 

def minforce(F, Z,  T, Min_Force=2):
    """Removes the data below minimum force given"""
    Mask = F > Min_Force
    Z = Z[Mask]
    F = F[Mask]
    T = T[Mask]
    return F, Z, T

def maxextention(F, Z, T, Max_Extension):   #Does not work yet
    """Removes the data above maximum extension given"""
    Mask = Z < Max_Extension
    Z = Z[Mask]
    F = F[Mask]
    T = T[Mask]
    return F, Z ,T


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