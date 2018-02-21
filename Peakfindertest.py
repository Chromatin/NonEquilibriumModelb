# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:52:49 2018

@author: nhermans
"""
import os #filenames
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
#from scipy.special import erf
import Functions as func
import Tools
import peakdetect as pk

folder = 'N:\\Rick\\Tweezer data\\2018_02_19_15x167 DNA\\data_013_Fit' #folder with chromosome sequence files (note, do not put other files in this folder)
filenames = os.listdir(folder)
os.chdir( folder )

Select=0 #1 for Selected Data, 0 for all data
Pulling = 1 #1 for only pulling data
DelBreaks =1 # 1 for deleting data after tether breaks
MinForce=2.5 #only analyze data above this force
MinZ, MaxZ = 0, True
Fmax_Hook=10
Err=5    #
steps , stacks = [],[] #used to save data
plt.close() #close all references to figures still open

for Filename in filenames:
    if Filename[-4:] != '.fit' :
        continue
    Headers,Data = Tools.read_data(Filename)
    LogFile=Tools.read_log(Filename[:-4]+'.log')
    Lc = float(Tools.find_param(LogFile,'L DNA (bp)')) # contour length (bp)
    p = float(Tools.find_param(LogFile,'p DNA (nm)') )  # persistence length (nm)p DNA (nm)
    S = float(Tools.find_param(LogFile,'S DNA (pN)') ) # stretch modulus (pN) S DNA (pN)
    k = float(Tools.find_param(LogFile,'k folded (pN/nm)') ) # Stiffness of the fiber, in pN/nm => k folded (pN/nm)
    N_tot = float(Tools.find_param(LogFile,'N nuc') )#number of nucleosomes N nuc
    N_tetra = float(Tools.find_param(LogFile,'N unfolded [F0]'))
    NRL = float(Tools.find_param(LogFile,'NRL (bp)') )#NRL (bp
    DNAds =  0.34 # rise per basepair (nm)
    kBT = 4.2 #pn/nm 
    Lmin=Lc-(N_tot-N_tetra)*NRL-N_tetra*80 # DNA handles in bp
    if Lmin <0: 
        print('<<<<<<<< warning: ',Filename, ': bad fit >>>>>>>>>>>>')
        continue
    Lmax=Lc-(N_tot)*80 # Max Z of the "beads on a string" conformation in bp
    Z_fiber = 1 * N_tot #Length of fiber in nm 
    print(Lmin,Lmax,N_tot, Filename)
    if MaxZ == True: MaxZ = (Lc+200)*DNAds
    Force = np.array([])
    Time=np.array([])
    Z=np.array([])
    Z_Selected=np.array([])
    
    for idx,item in enumerate(Data):                #Get all the data from the fitfile
        Force=np.append(Force,float(item.split()[Headers.index('F (pN)')]))
        Time=np.append(Time,float(item.split()[Headers.index('t (s)')]))
        Z_Selected=np.append(Z_Selected,float(item.split()[Headers.index('selected z (um)')])*1000)
        Z=np.append(Z,float(item.split()[Headers.index('z (um)')])*1000)
    
    #### This part removes all datapoints that should not be fitted 
    if Select == 1:                                 #If only the selected column is use do this
        ForceSelected = np.delete(Force, np.argwhere(np.isnan(Z_Selected)))
        Z_Selected=np.delete(Z, np.argwhere(np.isnan(Z_Selected)))
    if Select==0:
        ForceSelected=Force
        Z_Selected=Z
    if Pulling ==1: ForceSelected,Z_Selected = func.removerelease(ForceSelected,Z_Selected)
    if DelBreaks ==1: ForceSelected,Z_Selected = func.breaks(ForceSelected,Z_Selected, 1000)
    if MinForce > 0: ForceSelected,Z_Selected=func.minforce(ForceSelected,Z_Selected,MinForce)
    Z_Selected, ForceSelected = func.minforce(Z_Selected, ForceSelected, MinZ) #remove data below Z=0
    if MaxZ==True: Z_Selected, ForceSelected = func.minforce(-Z_Selected, ForceSelected, -Lc*DNAds*1.1) #remove data above Z=1.1*LC
    
    #Generate FE curves for possible states
    PossibleStates = np.arange(Lmin-200,Lc+50,1) #range to fit 
    dF=0.1 #Used to calculate local stiffness
    ProbSum=np.array([])
    for x in PossibleStates:
        Ratio=func.ratio(Lmin,Lmax,x)
        StateExtension=np.array(func.wlc(ForceSelected,p,S)*x*DNAds + func.hook(ForceSelected,k,Fmax_Hook)*Ratio*Z_fiber)
        StateExtension_dF=np.array(func.wlc(ForceSelected+dF,p,S)*x*DNAds + func.hook(ForceSelected+dF,k,Fmax_Hook)*Ratio*Z_fiber)
        LocalStiffness = np.subtract(StateExtension_dF,StateExtension)*(kBT) / dF # fix the units of KBT (pN nm -> pN um)
        DeltaZ=abs(np.subtract(Z_Selected,StateExtension))+Err
        std=np.divide(DeltaZ,np.sqrt(LocalStiffness))
        Pz=np.array((1-func.erfaprox(std))*np.sqrt(ForceSelected))
        ProbSum=np.append(ProbSum,np.sum(Pz)) 
    PeakInd,Peak=func.findpeaks(ProbSum, 25)
    Peaks = signal.find_peaks_cwt(ProbSum, np.arange(2.5,30), max_distances=np.linspace(75,75,len(ProbSum))) #numpy peakfinder, finds too many peaks, not used plot anyway

    #Peaks = pk.peakdetect(ProbSum, PossibleStates, 42)[0]
    #Peaks = np.array(Peaks)
    #Peaks=Peaks.astype(int)
    
    States=PossibleStates[PeakInd]
    #States2 = Peaks[:,0]
    
    Unwrapsteps=[]
    Stacksteps=[]
    for x in States:
        if x >= Lmax:
            Unwrapsteps.append(x)
        else:
            Stacksteps.append(x)
    Stacksteps=func.state2step(Stacksteps)
    Unwrapsteps=func.state2step(Unwrapsteps)
    if len(Unwrapsteps)>0: steps.extend(Unwrapsteps)
    if len(Stacksteps)>0: stacks.extend(Stacksteps)
    Tools.write_data('AllSteps.txt',Unwrapsteps,Stacksteps)
    
    #plotting
    # this plots the FE curve
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)
    fig1.suptitle(Filename, y=1)
    ax1.set_xlabel('Extension [nm]'), ax2.set_xlabel('Free basepairs')
    ax1.set_ylabel('Force [pN]'), ax2.set_ylabel('Probability [AU]')
    ax1.scatter(Z,Force, color="grey",s=1)
    ax1.scatter(Z_Selected,ForceSelected, color="blue", s=1)
    #ax2.set_xlim([0,Lc+50])    
    ax2.plot(PossibleStates,ProbSum)
    ax2.scatter(PossibleStates[(PeakInd)],Peak)
    #ax2.scatter(Peaks[:,0],Peaks[:,1], color="orange")
    #ax1.set_xlim([-100,Lc/2.8])
    #ax1.set_ylim([-4,25])

    #plotting
    # this plots the Timetrace    
    fig2 = plt.figure()    
    ax3 = fig2.add_subplot(1, 2, 1)
    ax4 = fig2.add_subplot(1, 2, 2)
    fig2.suptitle(Filename, y=1)
    ax3.set_xlabel('time [sec]'), ax4.set_xlabel('Probability [AU]')
    ax3.set_ylabel('Extension [bp nm]')
    #ax3.set_ylim([0,Lc*DNAds+200*DNAds])
    ax3.scatter(Time,Z, s=1)
    ax4.plot(ProbSum,PossibleStates*DNAds)
    ax4.scatter(Peak,PossibleStates[(PeakInd)]*DNAds, s=1)
    ax4.legend(label=States)
    
    for x in States:
        Ratio=func.ratio(Lmin,Lmax,x)
        Fit=np.array(func.wlc(Force,p,S)*x*DNAds + func.hook(Force,k,Fmax_Hook)*Ratio*Z_fiber)
        ax1.plot(Fit,Force, alpha=0.5, linestyle='-.')
        ax3.plot(Time,Fit, alpha=0.5, linestyle='-.')
        
        """ 
        for x in States2:
        Ratio=func.ratio(Lmin,Lmax,x)
        Fit=np.array(func.wlc(Force,p,S)*x*DNAds + func.hook(Force,k,Fmax_Hook)*Ratio*Z_fiber)
        ax1.plot(Fit,Force, linestyle=':')
        ax3.plot(Time,Fit, linestyle=':')
        """    

    fig1.tight_layout()
    fig1.savefig(Filename[0:-4]+'FoEx_all.png', dpi=800)
    fig1.show()
    fig2.tight_layout()
    fig2.savefig(Filename[0:-4]+'Time_all.png', dpi=800)    
    fig2.show()

#Stepsize,Sigma=func.fit_pdf(steps)
fig3 = plt.figure()
ax5 = fig3.add_subplot(1,1,1)
ax5.hist(steps,  bins = 50, range = [50,250] )
ax5.hist(stacks, bins = 50, range = [50,250])
ax5.set_xlabel('stepsize (bp)')
ax5.set_ylabel('Count')
ax5.set_title("Histogram stepsizes in bp")
ax5.legend(['25 nm steps', 'Stacking transitions'])
fig3.savefig('hist.png')
fig3.show()
