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
    par = Parameters()
    par.add('L_bp', value=3040)
    par.add('P_nm', value=50)
    par.add('S_pN', value=1000)
    par.add('degeneracy', value=0)
    par.add('z0_nm', value=0)
    par.add('N_tot', value=0)
    par.add('N4', value=0)
    par.add('k0_pN_nm', value=1)
    par.add('G1_kT', value=3)
    par.add('G2_kT', value=4)
    par.add('DNAds_nm',value= 0.34) # rise per basepair (nm)
    par.add('kBT_pN_nm',vaule= 4.2) #pn/nm 
    return par

def log_pars(LogFile):
    par = Parameters()
    par.add('L_bp', value=float(find_param(LogFile, 'L DNA (bp)')))
    par.add('P_nm', value=float(find_param(LogFile, 'p DNA (nm)')))
    par.add('S_pN', value=float(find_param(LogFile, 'S DNA (pN)')))
    par.add('degeneracy', value=0)
    par.add('z0_nm', value=2)
    par.add('k0_pN_nm', value=float(find_param(LogFile, 'k folded (pN/nm)')))
    par.add('N_tot', value=float(find_param(LogFile, 'N nuc')))
    par.add('N4', value=float(find_param(LogFile, 'N unfolded [F0]')))
    par.add('NRL_bp', value=float(find_param(LogFile, 'NRL (bp)')))
    par.add('G1_kT', value=3)
    par.add('G2_kT', value=4)
    par.add('DNAds_nm',value= 0.34) # rise per basepair (nm)
    par.add('kBT_pN_nm',vaule= 4.2) #pn/nm 
    return par

#LogFile = ReadLog("D:\\Klaas\\Tweezers\\Reconstituted chromatin\\ChromState\\2017_10_20_167x15\\Analysis\\15x167 FC1_data_006_40.log")
#Lc=FindParam(LogFile,"N nuc")