# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:52:17 2018

@author: nhermans
"""
#from lmfit import Parameters

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

##Open .dat/.fit files from magnetic tweezers
def read_data(Filename):
    f = open(Filename, 'r')
    #get headers
    headers=f.readlines()[0]
    headers=headers.split('\t')
    #get data
    f.seek(0) #seek to beginning of the file
    data=f.readlines()[1:]
    f.close()
    Force = np.array([])
    Time=np.array([])
    Z=np.array([])
    Z_Selected=np.array([])
    for idx,item in enumerate(data):                #Get all the data from the fitfile
        Force=np.append(Force,float(item.split()[headers.index('F (pN)')]))
        Time=np.append(Time,float(item.split()[headers.index('t (s)')]))
        Z_Selected=np.append(Z_Selected,float(item.split()[headers.index('selected z (um)')])*1000)
        Z=np.append(Z,float(item.split()[headers.index('z (um)')])*1000)
    return Force,Time,Z,Z_Selected

def read_log(Filename):
    f = open(Filename, 'r')
    lines=f.readlines()
    f.close()
    return lines

def find_param(Logfile, Param):
    for lines in Logfile:
        P =lines.split(' = ')
        if P[0]==Param:
            return P[1].strip('\n')
    return "no such parameter"
            
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

def plot_fe(f_array,z_array, units = 'nm'):
    plt.clf()    
    plt.scatter(z_array, f_array)
    plt.xlabel('{} {}'.format('extension', units))
    plt.ylabel('Force (pN)')
    plt.show()

def default_pars():
    """Default fitting parameters, returns a {dict} with 'key'= paramvalue"""
    par = {}
    par['L_bp']=3040
    par['P_nm'] =50
    par['S_pN'] =1000
    par['degeneracy'] =0
    par['z0_nm'] =0
    par['N_tot'] =0
    par['N4'] =0
    par['NRL_bp'] =167
    par['k_pN_nm'] =1
    par['G1_kT'] =3
    par['G2_kT'] =4
    par['DNAds_nm']= 0.34 # rise per basepair (nm)
    par['kBT_pN_nm']= 4.2 #pn/nm 
    par['Innerwrap_bp'] = 79 #number of basepairs in the inner turn wrap
    par['Fiber0_bp']=par['L_bp']-(par['N_tot']*par['Innerwrap_bp'])  #Transition between fiber and beats on a string
    par['LFiber_bp']=(par['N_tot']-par['N4'])*(par['NRL_bp']-par['Innerwrap_bp'])  #total number of bp in the fiber
    par['FiberStart_bp']=par['Fiber0_bp']-par['LFiber_bp'] #DNA handles
    return par

def log_pars(LogFile):
    """Reads in parameters from the logfile generate by the labview fitting program, returns a {dict} with 'key'= paramvalue"""
    par = {}
    par['L_bp'] =float(find_param(LogFile, 'L DNA (bp)'))
    par['P_nm'] =float(find_param(LogFile, 'p DNA (nm)'))
    par['S_pN'] =float(find_param(LogFile, 'S DNA (pN)'))
    par['degeneracy'] =0
    par['z0_nm'] =2
    par['k_pN_nm'] =float(find_param(LogFile, 'k folded (pN/nm)'))
    par['N_tot'] =float(find_param(LogFile, 'N nuc'))
    par['N4'] =float(find_param(LogFile, 'N unfolded [F0]'))
    par['NRL_bp'] =float(find_param(LogFile, 'NRL (bp)'))
    par['ZFiber_nm']=float(find_param(LogFile, 'l folded (nm)'))
    par['G1_kT'] =3
    par['G2_kT'] =4
    par['DNAds_nm']= 0.34 # rise per basepair (nm)
    par['kBT_pN_nm']= 4.2 #pn/nm 
    par['Innerwrap_bp'] = 79 #number of basepairs in the inner turn wrap
    par['Fiber0_bp']=par['L_bp']-(par['N_tot']*par['Innerwrap_bp'])  #Transition between fiber and beats on a string
    par['LFiber_bp']=(par['N_tot']-par['N4'])*(par['NRL_bp']-par['Innerwrap_bp'])  #total number of bp in the fiber
    par['FiberStart_bp']=par['Fiber0_bp']-par['LFiber_bp']
    return par

def handle_data(Force,Z,Z_Selected, Handles, Pars= default_pars(), Window=5):
    """Reads in parameters from the logfile generate by the labview fitting program"""
    if Handles['Select'] == 1:                                 #If only the selected column is use do this
        ForceSelected = np.delete(Force, np.argwhere(np.isnan(Z_Selected)))
        Z_Selected=np.delete(Z, np.argwhere(np.isnan(Z_Selected)))
    if len(Z_Selected)==0: 
        print('==> Nothing Selected!')
        return [], []
    if Handles['Select']==0:
        ForceSelected=Force
        Z_Selected=Z
    if Handles['Pulling']: ForceSelected,Z_Selected = removerelease(ForceSelected,Z_Selected)
    if Handles['DelBreaks']: ForceSelected,Z_Selected = breaks(ForceSelected,Z_Selected, 1000)
    if Handles['MinForce'] > 0: ForceSelected,Z_Selected = minforce(ForceSelected,Z_Selected,Handles['MinForce'])
#    Z_Selected, ForceSelected = func.minforce(Z_Selected, ForceSelected, MinZ) #remove data below Z=0
    if Handles['MaxZ'] == True:  #Remove all datapoints after max extension
        Handles['MaxZ'] = (Pars['L_bp']+100)*Pars['DNAds_nm']
        Z_Selected, ForceSelected = minforce(Z_Selected, ForceSelected, - Pars['L_bp']*Pars['DNAds_nm']*1.1) #remove data above Z=1.1*LC
    if Handles['Denoise']: Z_Selected = signal.medfilt(Z_Selected,Window)
    return Z_Selected, ForceSelected

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
#LogFile = ReadLog("D:\\Klaas\\Tweezers\\Reconstituted chromatin\\ChromState\\2017_10_20_167x15\\Analysis\\15x167 FC1_data_006_40.log")
#Lc=FindParam(LogFile,"N nuc")