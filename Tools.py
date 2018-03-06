# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:52:17 2018

@author: nhermans
"""
#from lmfit import Parameters

import matplotlib.pyplot as plt
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
    return headers,data

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
    par = {}
    par['L_bp']=3040
    par['P_nm'] =50
    par['S_pN'] =1000
    par['degeneracy'] =0
    par['z0_nm'] =0
    par['N_tot'] =0
    par['N4'] =0
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

#LogFile = ReadLog("D:\\Klaas\\Tweezers\\Reconstituted chromatin\\ChromState\\2017_10_20_167x15\\Analysis\\15x167 FC1_data_006_40.log")
#Lc=FindParam(LogFile,"N nuc")