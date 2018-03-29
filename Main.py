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

folder = r'N:\Rick\Tweezer data\Pythontestfit\ProbabilityTest'
folder = folder.replace('\\', '\\\\')                                           #Replaces \ for \\

filenames = os.listdir(folder)
os.chdir(folder)

Handles = Tools.Define_Handles(Select=True)
steps , stacks = [],[]                                                          #used to save data (T-test)
Steps , Stacks = [],[]                                                          #used to save data (Smoothening)
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
    print(int(Pars['N_tot']), "Nucleosomes in", Filename, "( Fig.", Fignum, "&", Fignum+1, ")")

    #Remove all datapoints that should not be fitted
    Z_Selected, F_Selected, T_Selected = Tools.handle_data(Force, Z, Time, Z_Selected, Handles, Pars)

    if len(Z_Selected)<10:  
        print("<<<<<<<<<<<", Filename,'==> No data points left after filtering!>>>>>>>>>>>>')
        continue

    Filename = Filename.replace('_', '\_')                                      #Right format for the plot headers
    
    PossibleStates = np.arange(Pars['FiberStart_bp']-200, Pars['L_bp']+50,1)
    ProbSum = func.probsum(F_Selected, Z_Selected, PossibleStates, Pars) 
    PeakInd, Peak = func.findpeaks(ProbSum, 25)                                 #Find Peaks    
    Starting_States = PossibleStates[PeakInd]                                   #Defines state for each peak
    States=func.find_states_prob(F_Selected,Z_Selected,Pars, MergeStates=True, P_Cutoff=0.1) #Finds States
    AAA = func.STD(F_Selected, Z_Selected, PossibleStates, Pars)
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
    ax2 = fig1.add_subplot(1, 2, 2)
    fig1.suptitle(Filename, y=.99)
    ax1.set_title("Extension-Force Curve")
    ax2.set_title(" ")
    ax1.set_xlabel(r'\textbf{Force} (pN)'), ax2.set_ylabel(r'\textbf{Probability} (AU)')
    ax1.set_ylabel(r"\textbf{Extension} (nm)"), ax2.set_xlabel(r"\textbf{Free base pair} (nm)") #(nm) should be removed
    ax1.scatter(Force, Z, c=Time, cmap='gray', lw=0.1, s=5)
 #   ax1.scatter(F_Selected, Z_Selected, color="blue", s=1)   
    ax2.plot(PossibleStates,ProbSum, alpha=0.1)
    ax2.scatter(PossibleStates[(PeakInd)],Peak, alpha=0.1)
    ax1.set_xlim([np.min(F_Selected)-0.1*np.max(F_Selected), np.max(F_Selected)+0.1*np.max(F_Selected)])
    ax1.set_ylim([np.min(Z_Selected)-0.1*np.max(Z_Selected), np.max(Z_Selected)+0.1*np.max(Z_Selected)])
    #ax2.set_xlim([np.min(PossibleStates)*Pars['DNAds_nm'], np.max(PossibleStates)*Pars['DNAds_nm']+0.1*np.max(PossibleStates)]*Pars['DNAds_nm'])


    # this plots the Timetrace
    fig2 = plt.figure()
    ax3 = fig2.add_subplot(1, 2, 1)
    ax4 = fig2.add_subplot(1, 2, 2, sharey=ax3)
    fig2.suptitle(Filename, y=.99)
    ax3.set_title("Timetrace Curve")
    ax4.set_title(" ")
    ax3.set_xlabel(r'\textbf{Time} (s)'), ax4.set_xlabel(r'\textbf{Probability} (AU)')
    ax3.set_ylabel(r'\textbf{Extension} (bp nm)')
    ax3.set_ylim([0, Pars['L_bp']*Pars['DNAds_nm']+100])
    ax3.scatter(Time,Z,  c=Time, cmap='gray', lw=0.1, s=5)
    ax3.scatter(T_Selected, Z_Selected, color='blue', s=1)
    ax4.plot(ProbSum,PossibleStates*Pars['DNAds_nm'], alpha=0.1)
    ax4.scatter(Peak,PossibleStates[(PeakInd)]*Pars['DNAds_nm'], color='blue', alpha=0.1)
    ax3.set_xlim([np.min(Time)-0.1*np.max(Time), np.max(Time)+0.1*np.max(Time)])
    ax3.set_ylim([np.min(Z)-0.1*np.max(Z), np.max(Z)+0.1*np.max(Z)])
    
##############################################################################################
######## Begin Smoothening
   
    Smoothness = 70
    SmoothProbSum = func.Conv(ProbSum,Smoothness)
    SmoothPeakInd, SmoothPeak = func.findpeaks(SmoothProbSum, 25)
    SmoothStates = PossibleStates[SmoothPeakInd]
    ax2.plot(PossibleStates, SmoothProbSum, 'g-', lw=2)
    ax2.scatter(PossibleStates[(SmoothPeakInd)],SmoothPeak, color='green')
    
    ax4.plot(SmoothProbSum, PossibleStates*Pars['DNAds_nm'], color='green')
    ax4.scatter(SmoothPeak,PossibleStates[(SmoothPeakInd)]*Pars['DNAds_nm'], color='green')
    
    colors = [plt.cm.brg(each) for each in np.linspace(0, 1, len(SmoothStates))]#Color pattern for the states
    dX = 10                                                                     #Offset for text in plot
    
    #Plot the states
    for i, col in zip(enumerate(SmoothStates), colors):
        Ratio = func.ratio(i[1],Pars)
        Fit = np.array(func.wlc(Force,Pars)*i[1]*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        ax1.plot(Force, Fit, alpha=0.9, linestyle=':', color=tuple(col))
        ax2.vlines(i[1], 0, SmoothPeak[i[0]], linestyle=':', color=tuple(col))
        ax2.text(i[1], SmoothPeak[i[0]]+dX, int(i[1]), fontsize=8, horizontalalignment='center')
        ax3.plot(Time,Fit, alpha=0.9, linestyle=':', color=tuple(col))
        ax4.hlines(i[1]*Pars['DNAds_nm'], 0, SmoothPeak[i[0]], color=tuple(col), linestyle=':')
        ax4.text(SmoothPeak[i[0]]+dX, i[1]*Pars['DNAds_nm'], int(i[1]*Pars['DNAds_nm']), fontsize=8, verticalalignment='center')
    
    Statemask = func.attribute2state(F_Selected,Z_Selected,SmoothStates, Pars)
    
    #Plot datapoint in the right color
    for i, col in zip(enumerate(Statemask), colors):
        ax1.scatter(F_Selected[Statemask==i[0]], Z_Selected[Statemask==i[0]], color=tuple(col), s=5)
        ax3.scatter(T_Selected[Statemask==i[0]], Z_Selected[Statemask==i[0]], color=tuple(col), s=5)   

    Unwrapsteps = []
    Stacksteps = []
    for x in SmoothStates:
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

    Fignum += 2


#Stepsize,Sigma=func.fit_pdf(steps)
fig3 = plt.figure()
ax5 = fig3.add_subplot(1,2,1)
ax6 = fig3.add_subplot(1,2,2, sharey=ax5)
ax5.hist(steps,  bins = 50, range = [0,400], lw=0.5, color='blue', label='25 nm steps')
ax5.hist(stacks, bins = 50, range = [0,400], lw=0.5, color='orange', label='Stacking transitions')
ax6.hist(Steps,  bins = 50, range = [0,400], lw=0.5, color='blue', label='25 nm steps')
ax6.hist(Stacks, bins = 50, range = [0,400], lw=0.5, color='orange', label='Stacking transitions')
ax5.set_xlabel('stepsize (bp)')
ax5.set_ylabel('Count')
ax5.set_title("Histogram stepsizes in bp using T-test")
ax5.legend(loc='best')
ax6.set_xlabel('stepsize (bp)')
ax6.set_ylabel('Count')
ax6.set_title("Histogram stepsizes in bp using smoothening")
ax6.legend(loc='best')
fig3.tight_layout()
#fig3.savefig('hist.png')
