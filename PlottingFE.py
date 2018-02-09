# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:56:10 2018

@author: nhermans
"""

import matplotlib.pyplot as plt
import numpy as np
import Functions as func
import Tools

Select=1 #1 for Selected Data, 0 for all data
Pulling = 1 #1 for only pulling data
DelBreaks = 0 # 1 for deleting data after tether breaks
MinForce=1 #only analyze data above this force

Filename="D:\\Klaas\\Tweezers\\Python Scripts\\NonEquilibriumModel\\15x167 FC1_data_006_40_steps.fit"
Headers,Data = Tools.read_data(Filename)
LogFile=Tools.read_log(Filename[:-10]+'.log')
Lc = float(Tools.find_param(LogFile,'L DNA (bp)')) # contour length (bp)
p = float(Tools.find_param(LogFile,'p DNA (nm)') )  # persistence length (nm)p DNA (nm)
S = float(Tools.find_param(LogFile,'S DNA (pN)') ) # stretch modulus (pN) S DNA (pN)
k=1 #k = float(Tools.find_param(LogFile,'k folded (pN/nm)') ) # Stiffness of the fiber, in pN/nm => k folded (pN/nm)
N = float(Tools.find_param(LogFile,'N nuc') )#number of nucleosomes N nuc
N_tetra = float(Tools.find_param(LogFile,'N unfolded [F0]'))
NRL = float(Tools.find_param(LogFile,'NRL (bp)') )#NRL (bp
DNAds =  0.34 # rise per basepair (nm)
kBT = 4.2 #pn/nm 
Lmin=Lc-N*NRL+N_tetra*75 # DNA handles in bp
Lmax=Lc-(N+N_tetra)*75 # Max Z of the "beads on a string" conformation in bp
Z_fiber = 1 * N #Length of fiber in nm 
print(Lmin,Lmax,Lc, Filename)

Force = np.array([])
Time=np.array([])
Z=np.array([])
Z_Selected=np.array([])

for idx,item in enumerate(Data):                #Get all the data from the fitfile
    Force=np.append(Force,float(item.split()[Headers.index('F (pN)')]))
    Time=np.append(Time,float(item.split()[Headers.index('t (s)')]))
    Z_Selected=np.append(Z_Selected,float(item.split()[Headers.index('selected z (um)')])*1000)
    Z=np.append(Z,float(item.split()[Headers.index('z (um)')])*1000)
if Select == 1:                                 #If only the selected column is use do this
    ForceSelected = np.delete(Force, np.argwhere(np.isnan(Z_Selected)))
    Z_Selected=np.delete(Z, np.argwhere(np.isnan(Z_Selected)))
if Select==0:
    ForceSelected=Force
    Z_Selected=Z
if Pulling ==1: ForceSelected,Z_Selected = func.removerelease(ForceSelected,Z_Selected)
if DelBreaks ==1: ForceSelected,Z_Selected = func.breaks(ForceSelected,Z_Selected)
if MinForce > 0: ForceSelected,Z_Selected=func.minforce(ForceSelected,Z_Selected,MinForce)

#ForceSelected=ForceSelected[1145:1155]
#Z_Selected=Z_Selected[1145:1155]

plt.figure(1)
plt.xlabel('Extension [nm]')
plt.ylabel('Force [pN]')
plt.scatter(Z, Force, color='grey',alpha=0.3, marker='+')
plt.scatter(Z_Selected,ForceSelected)

#Generate FE curves for possible states
#PossibleStates = np.arange(Lc-120,Lc,1) #range to fit
PossibleStates = np.arange(Lmin-200,Lc+50,1)#range to fit 

dF=0.1 #Used to calculate local stiffness
ProbSum=np.array([])
for x in PossibleStates:
    Ratio=func.ratio(Lmin,Lmax,x)
    StateExtension=np.array([func.wlc(ForceSelected,p,S)*x*DNAds + func.hook(ForceSelected,k,10)*Ratio*Z_fiber])
    StateExtension_dF=np.array([func.wlc(ForceSelected+dF,p,S)*x*DNAds + func.hook(ForceSelected+dF,k,10)*Ratio*Z_fiber])
    LocalStiffness = np.subtract(StateExtension_dF,StateExtension)*(kBT) / dF # fix the units of KBT (pN nm -> pN um)
    DeltaZ=np.subtract(Z_Selected,StateExtension)
    std=abs(np.divide(DeltaZ,np.sqrt(LocalStiffness)))
    Pz=np.array([1-func.erfaprox(std)])
    ProbSum=np.append(ProbSum,np.sum(Pz)) 
#    Fit=np.array(func.wlc(Force,p,S)*x*DNAds + func.hook(Force,k,10)*Ratio*Z_fiber)
#    plt.plot(Fit,Force, linestyle=':')
PeakInd,Peak=func.findpeaks(ProbSum, 25)
#Peaks = signal.find_peaks_cwt(ProbSum, np.arange(5,30)) #numpy peakfinder, finds too many peaks, not used plot anyway
States=PossibleStates[PeakInd]
#Fit=np.array(func.wlc(Force,p,S)*(Lc-80)*DNAds + func.hook(Force,k,10)*Ratio*Z_fiber)
#plt.plot(Fit,Force, linestyle=':')
#plotting

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
plt.title(Filename)
ax1.set_xlabel('Extension [bp]'), 
ax1.set_ylabel('Force [pN]'), ax2.set_ylabel('Probability [AU]')
ax1.set_xlim([0,Lc+200])
ax1.scatter(Z/DNAds,Force)
ax2.plot(PossibleStates,ProbSum)
ax2.scatter(PossibleStates[(PeakInd)],Peak)
#ax2.scatter(ProbSum[(Peaks)],PossibleStates[(Peaks)])
for x in States:
    Ratio=func.ratio(Lmin,Lmax,x)
    Fit=np.array(func.wlc(Force,p,S)*x*DNAds + func.hook(Force,k,10)*Ratio*Z_fiber)
    plt.figure(2)
    ax1.plot(Fit/DNAds,Force, linestyle=':')
    plt.figure(1)
    plt.plot(Fit,Force, linestyle=':')
plt.savefig(Filename[0:-4]+'_full.png')

plt.figure(3)
plt.scatter(PossibleStates, ProbSum)
plt.scatter(PossibleStates[(PeakInd)],Peak)
plt.xlabel('Extension [bp]')
plt.ylabel('Probability [AU]')
plt.show()