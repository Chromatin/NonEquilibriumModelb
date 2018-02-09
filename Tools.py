# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:52:17 2018

@author: nhermans
"""
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
#LogFile = ReadLog("D:\\Klaas\\Tweezers\\Reconstituted chromatin\\ChromState\\2017_10_20_167x15\\Analysis\\15x167 FC1_data_006_40.log")
#Lc=FindParam(LogFile,"N nuc")