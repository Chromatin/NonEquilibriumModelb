# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:36:15 2018

@author: nhermans
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import Functions as func
import Tools

Filename="D:\\Klaas\\Tweezers\\Reconstituted chromatin\\ChromState\\2017_10_20_167x15\\Analysis\\15x167 FC1_data_006_40_stepsandHO"
Select=1 #1 for Selected Data, 0 for all data
Pulling = 0 #1 for only pulling data

LogFile=Tools.read_log(Filename+'.log')
Lc = float(Tools.find_param(LogFile,'L DNA (bp)')) # contour length (bp)
p = float(Tools.find_param(LogFile,'p DNA (nm)') )  # persistence length (nm)p DNA (nm)
S = float(Tools.find_param(LogFile,'S DNA (pN)') ) # stretch modulus (pN) S DNA (pN)
k = float(Tools.find_param(LogFile,'k folded (pN/nm)') ) # Stiffness of the fiber, in pN/nm => k folded (pN/nm)
N = float(Tools.find_param(LogFile,'N nuc') )#number of nucleosomes N nuc
N_tetra = float(Tools.find_param(LogFile,'N unfolded [F0]'))
NRL = float(Tools.find_param(LogFile,'NRL (bp)') )#NRL (bp
DNAds =  0.34 # rise per basepair (nm)
kBT = 4.2 #pn/nm 
Z_fiber = 1 * N #Length of fiber in nm 
Lmin=Lc-N*NRL # handles
Lmax=Lc-N*75 # Beads on a string
Headers,Data = Tools.read_data(Filename+'.fit')
print(Lmin,Lmax,Lc)
Force = np.array([])
Time=np.array([])
Z=np.array([])
Z_Selected=np.array([])

for idx,item in enumerate(Data):
    Force=np.append(Force,float(item.split()[Headers.index('F (pN)')]))
    Time=np.append(Time,float(item.split()[Headers.index('t (s)')]))
    Z_Selected=np.append(Z_Selected,float(item.split()[Headers.index('selected z (um)')])*1000)
    Z=np.append(Z,float(item.split()[Headers.index('z (um)')])*1000)

#Generate FE curves for possible states
PossibleStates = np.arange(Lmin-500,Lc+10,1)
dF=0.1   #Used to calculate local stiffness
ProbSum=np.array([])

if Select == 1:
    ForceSelected = np.delete(Force, np.argwhere(np.isnan(Z_Selected)))
    Z_Selected=np.delete(Z, np.argwhere(np.isnan(Z_Selected)))
if Select==0:
    ForceSelected=Force
    Z_Selected=Z
if Pulling ==1:
    test=0
    Pullingtest=np.array([])
    for i,x in enumerate(ForceSelected): #werkt nog niet
        if x < test:
           Pullingtest= np.append(Pullingtest,i)
        test=x
    ForceSelected=np.delete(ForceSelected, Pullingtest)
    Z_Selected=np.delete(Z_Selected,Pullingtest)    

for x in PossibleStates:
    Ratio=func.ratio(Lmin,Lmax,x)
    StateExtension=np.array([func.wlc(ForceSelected,p,S)*x*DNAds + func.hook(ForceSelected,k,10)*Ratio*Z_fiber])
    StateExtension_dF=np.array([func.wlc(ForceSelected+dF,p,S)*x*DNAds + func.hook(ForceSelected+dF,k,10)*Ratio*Z_fiber])
    LocalStiffness = np.subtract(StateExtension_dF,StateExtension)*(kBT) / dF # fix the units of KBT (pN nm -> pN um)
    DeltaZ=np.subtract(Z_Selected,StateExtension)
    std=abs(np.divide(DeltaZ,np.sqrt(LocalStiffness)))
    Pz=np.array([1-func.erfaprox(std)])
    ProbSum=np.append(ProbSum,np.sum(Pz)) 
PeakInd,Peak=func.findpeaks(ProbSum)
Peaks = signal.find_peaks_cwt(ProbSum, np.arange(5,30))

States=PossibleStates[PeakInd]
Dist=func.state2step(States)

#Saving
#import fileio
#fileio.write_tdms(Filename, Data, States, pars={0})

#Plotting
plt.close()
plt.figure(1)
plt.xlabel('Extension [nm]')
plt.ylabel('Force [pN]')
plt.scatter(Z, Force)
plt.scatter(Z_Selected,ForceSelected)

#plt.figure(2)
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
plt.title(Filename)
ax1.set_xlabel('time [sec]'), ax2.set_xlabel('Probability [AU]')
ax1.set_ylabel('Extension [bp]')
ax1.set_ylim([0,Lc+200])
ax1.scatter(Time,Z/DNAds)
ax2.plot(ProbSum,PossibleStates)
ax2.scatter(Peak,PossibleStates[(PeakInd)])
ax2.scatter(ProbSum[(Peaks)],PossibleStates[(Peaks)])

for x in States:
    Ratio=func.ratio(Lmin,Lmax,x)
    Fit=np.array(func.wlc(Force,p,S)*x*DNAds + func.hook(Force,k,10)*Ratio*Z_fiber)
    plt.figure(2)
    ax1.plot(Time,Fit/DNAds, linestyle=':')
    plt.figure(1)
    plt.plot(Fit,Force, linestyle=':')
        
plt.show()