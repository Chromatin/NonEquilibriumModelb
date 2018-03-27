# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:52:49 2018

@author: nhermans & rrodrigues
"""
import os 
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import numpy as np
import Functions as func
import Tools
import pickle

folder = 'C:\\Users\\rmerc\\OneDrive\\Documenten\\Universiteit Leiden\\Bachelor Research\\Test\\20180327 ForceExtensionCurvefitting-Probability\\TestData'

filenames = os.listdir(folder)
os.chdir(folder)

Handles = Tools.Define_Handles()
steps, stacks = [],[]                                                           #used to save data (T-test)
Steps , Stacks = [],[]                                                          #used to save data (Smoothened)

Fignum = 1

plt.close('all')                                                                #Close all the figures from previous sessions

for Filenum, Filename in enumerate(filenames):
    if Filename[-4:] != '.fit' :
        continue
    Force, Time, Z, Z_Selected = Tools.read_data(Filename)                      #loads the data from the filename
    LogFile = Tools.read_log(Filename[:-4]+'.log')                              #loads the log file with the same name
    Pars = Tools.log_pars(LogFile)                                              #Reads in all the parameters from the logfile

    if Pars['FiberStart_bp'] <0: 
        print('<<<<<<<< warning: ',Filename, ': bad fit >>>>>>>>>>>>')
        continue
#    print(int(Pars['N_tot']), "Nucleosomes in", Filename, "( Fig.", Fignum, "&", Fignum+1, ")")

    #Remove all datapoints that should not be fitted
    Z_Selected, F_Selected, T_Selected = Tools.handle_data(Force, Z, Time, Z_Selected, Handles, Pars)

    if len(Z_Selected)<10:  
        print("<<<<<<<<<<<", Filename,'==> No data points left after filtering!>>>>>>>>>>>>')
        continue

    Filename = Filename.replace('_', '\_')                                      #Right format for the plot headers
    
    PossibleStates = np.arange(Pars['FiberStart_bp']-200, Pars['L_bp']+50,1)
    ProbSum = func.probsum(F_Selected, Z_Selected, PossibleStates, Pars) 
    PeakInd, Peak = func.findpeaks(ProbSum, 25)                                 #Find Peaks    
    Starting_States = PossibleStates[PeakInd]                                            #Defines state for each peak
    States=func.find_states_prob(F_Selected,Z_Selected,Pars, MergeStates=True, P_Cutoff=0.1) #Finds States
    
    #Calculates stepsize
    Unwrapsteps = []
    Stacksteps = []
    for x in States:
        if x >= Pars['Fiber0_bp']:
            Unwrapsteps.append(x)
        else:
            Stacksteps.append(x)
    Stacksteps = func.state2step(Stacksteps)
    Unwrapsteps = func.state2step(Unwrapsteps)
    if len(Unwrapsteps)>0: steps.extend(Unwrapsteps)
    if len(Stacksteps)>0: stacks.extend(Stacksteps)
    #Tools.write_data('AllSteps.txt',Unwrapsteps,Stacksteps)

    # this plots the Force-Extension curve
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2, sharex=ax1)
    fig1.suptitle(Filename, y=.99)
    ax1.set_title(" ")
    ax2.set_title(" ")
    ax1.set_xlabel(r"\textbf{Extension} (nm)"), ax2.set_xlabel(r"\textbf{Free base pair} (nm)") #(nm) should be removed
    ax1.set_ylabel(r'\textbf{Force} (pN)'), ax2.set_ylabel(r'\textbf{Probability} (AU)')
    ax1.scatter(Z,Force, c=Time, cmap='gray', lw=0.1, s=5)
#    ax1.scatter(Z_Selected,F_Selected, color="blue", s=1)   
    ax2.plot(0.34*PossibleStates,ProbSum, alpha=0.1)                            #*0.34 should be removed
    ax2.scatter(0.34*PossibleStates[(PeakInd)],Peak, alpha=0.1)                 #*0.34 should be removed
    ax1.set_xlim([np.min(Z_Selected)-0.1*np.max(Z_Selected), np.max(Z_Selected)+0.1*np.max(Z_Selected)])
    ax1.set_ylim([np.min(F_Selected)-0.1*np.max(F_Selected), np.max(F_Selected)+0.1*np.max(F_Selected)])
    #ax2.set_xlim([np.min(PossibleStates)*Pars['DNAds_nm'], np.max(PossibleStates)*Pars['DNAds_nm']+0.1*np.max(PossibleStates)]*Pars['DNAds_nm'])

#######################################################################################################################
######## Begin Smoothening
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    Smoothness = 70
    SmoothPeakInd, SmoothPeak = func.findpeaks(smooth(ProbSum,Smoothness), 25)
    ax2.plot(0.34*PossibleStates, smooth(ProbSum,Smoothness), 'g-', lw=2)       #*0.34 should be removed
    ax2.scatter(0.34*PossibleStates[(SmoothPeakInd)],SmoothPeak, color='green') #*0.34 should be removed
    
    colors = [plt.cm.brg(each) for each in np.linspace(0, 1, len(PossibleStates[SmoothPeakInd]))]

    for i, col in zip(enumerate(PossibleStates[SmoothPeakInd]), colors):
        Ratio = func.ratio(i[1],Pars)
        Fit = np.array(func.wlc(Force,Pars)*i[1]*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        ax1.plot(Fit,Force, alpha=0.9, linestyle=':', color=tuple(col))
        ax2.vlines(0.34*i[1], 0, SmoothPeak[i[0]], linestyle=':', color=tuple(col), )
    

    Statemask = func.attribute2state(F_Selected,Z_Selected,PossibleStates[SmoothPeakInd], Pars)
    
    for i, col in zip(enumerate(Statemask), colors):
        ax1.scatter(Z_Selected[Statemask==i[0]], F_Selected[Statemask==i[0]], color=tuple(col), s=5)   


    Unwrapsteps = []
    Stacksteps = []
    for x in PossibleStates[SmoothPeakInd]:
        if x >= Pars['Fiber0_bp']:
            Unwrapsteps.append(x)
        else:
            Stacksteps.append(x)
    Stacksteps = np.diff(np.array(Stacksteps))
    Unwrapsteps = np.diff(np.array(Unwrapsteps))
    if len(Unwrapsteps)>0: Steps.extend(Unwrapsteps)
    if len(Stacksteps)>0: Stacks.extend(Stacksteps)
    #Tools.write_data('AllSteps.txt',Unwrapsteps,Stacksteps)

#######################################################################################################################


    # this plots the Timetrace
    fig2 = plt.figure()
    ax3 = fig2.add_subplot(1, 2, 1)
    ax4 = fig2.add_subplot(1, 2, 2, sharey=ax3)
    fig2.suptitle(Filename, y=.99)
    ax3.set_title(" ")
    ax4.set_title(" ")
    ax3.set_xlabel(r'\textbf{Time} (s)'), ax4.set_xlabel(r'\textbf{Probability} (AU)')
    ax3.set_ylabel(r'\textbf{Extension} (bp nm)')
    ax3.set_ylim([0, Pars['L_bp']*Pars['DNAds_nm']+100])
    ax3.scatter(Time,Z,  c=Time, cmap='gray', lw=0.1, s=5)
    ax3.scatter(T_Selected, Z_Selected, color='blue', s=1)
    ax4.plot(ProbSum,PossibleStates*Pars['DNAds_nm'])
    ax4.scatter(Peak,PossibleStates[(PeakInd)]*Pars['DNAds_nm'], color='blue', s=1)
    ax3.set_xlim([np.min(Time)-0.1*np.max(Time), np.max(Time)+0.1*np.max(Time)])
    ax3.set_ylim([np.min(Z)-0.1*np.max(Z), np.max(Z)+0.1*np.max(Z)])

    for x in States:
        Ratio = func.ratio(x,Pars)
        Fit = np.array(func.wlc(Force,Pars)*x*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        ax1.plot(Fit,Force, alpha=0.1, linestyle='-.')
        ax3.plot(Time,Fit, alpha=0.1, linestyle='-.')
        
    

    Filename = Filename.replace('\_', '_')                                      #Right format for sa

    fig1.tight_layout()
    pickle.dump(fig1, open(Filename[0:-4]+'.FoEx_all.pickle', 'wb'))            #Saves the figure, so it can be reopend
    fig1.savefig(Filename[0:-4]+'FoEx_all.pdf')
    fig1.show()
    fig2.tight_layout()
    pickle.dump(fig2, open(Filename[0:-4]+'.Time_all.pickle', 'wb'))            #Saves the figure, so it can be reopend
    fig2.savefig(Filename[0:-4]+'Time_all.pdf')    
    fig2.show()

    Fignum += 1


fig3 = plt.figure()
ax5 = fig3.add_subplot(1,2,1)
ax6 = fig3.add_subplot(1,2,2, sharey=ax5)
ax5.hist(steps,  bins = 50, range = [0,400], label='25 nm steps')
ax5.hist(stacks, bins = 50, range = [0,400], label='Stacking transitions')
ax6.hist(Steps,  bins = 50, range = [0,400], label='25 nm steps')
ax6.hist(Stacks, bins = 50, range = [0,400], label='Stacking transitions')
ax5.set_xlabel('stepsize (bp)')
ax5.set_ylabel('Count')
ax5.set_title("Histogram stepsizes in bp using T-test")
ax5.legend()
ax6.set_xlabel('stepsize (bp)')
ax6.set_ylabel('Count')
ax6.set_title("Histogram stepsizes in bp using smoothening")
ax6.legend()
fig3.tight_layout()
#fig3.savefig('hist.png')
