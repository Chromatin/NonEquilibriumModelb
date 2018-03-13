# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:52:49 2018

@author: nhermans
"""
import os 
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import Functions as func
import Tools

folder = 'N:\\Rick\\Tweezer data\\Pythontestfit' #folder with chromosome sequence files (note, do not put other files in this folder)
filenames = os.listdir(folder)
os.chdir(folder)

Handles = Tools.Define_Handles()
steps , stacks = [],[]                                                          #used to save data
Fignum = 1

plt.close('all')                                                                #Close all the figures from previous sessions

for Filename in filenames:
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

    #Generate FE curves for possible states
    PossibleStates = np.arange(Pars['FiberStart_bp']-200, Pars['L_bp']+50,1)    #range to fit 
    ProbSum = func.probsum(F_Selected, Z_Selected, PossibleStates, Pars)     #Calculate probability landscape
    PeakInd, Peak = func.findpeaks(ProbSum, 25)                                 #Find Peaks
    States = PossibleStates[PeakInd]                                            #Defines state for each peak
    
    """The numpy peakfinder may find more peaks"""
    #Peaks = signal.find_peaks_cwt(ProbSum, np.arange(2.5,30))                  #numpy peakfinder    
    #States=PossibleStates[Peaks]
        
    #Calculate for each datapoint which state it most likely belongs too 
    StateMask = func.attribute2state(F_Selected,Z_Selected,States,Pars)
    
    #Remove states with 5 or less datapoints
    RemoveStates = func.removestates(StateMask)
    if len(RemoveStates)>0:
        States = np.delete(States, RemoveStates)
        StateMask = func.attribute2state(F_Selected, Z_Selected, States, Pars)
    
    # Merging states that are have similar mean/variance according to Welch test
    UnMergedStates = States                                                     #Used to co-plot the initial states found
    if len(States) > 1:
        MergeStates = True
    else:
        MergeStates = False
    
    P_Cutoff = 0.05                                                             #Significance for merging states    
    
    while MergeStates:                                                          #remove states untill all states are significantly different
        T_test = np.array([])
        CrossCor = np.array([])                                                 #array for Crosscorr values comparing different states
        for i,j in enumerate(States):
            if i > 0:               
                Prob = stats.ttest_ind((StateMask==i)*Z_Selected,(StateMask==i-1)*Z_Selected, equal_var=False) #get two arrays for t_test
                T_test = np.append(T_test,Prob[1])                             #Calculates the p-value of neighboring states with Welch test

        if len(T_test)==0: 
            MergeStates = False            
            continue
                      
        #Merges states that are most similar, and are above the p_cutoff minimal significance t-test value
        HighP = np.argmax(T_test)
        if T_test[HighP] > P_Cutoff:                                            #Merge the highest p-value states
            if sum((StateMask == HighP + 1) * 1) < sum((StateMask == HighP) * 1): 
                DelState = HighP + 1
            else:
                DelState = HighP
            States = np.delete(States, DelState)                                #deletes the state with the fewest datapoints from the state array
            StateMask = func.attribute2state(F_Selected, Z_Selected, States, Pars)
            Z_NewState = (StateMask == HighP) * Z_Selected                      #Get all the data for this state to recalculate mean    
        else:
            MergeStates = False  # Stop merging states
               
        #calculate the number of L_unrwap for the new state
        if MergeStates:
            #find value for merged state with gaus fit / mean
            StateProbSum = func.probsum(F_Selected[Z_NewState != 0],Z_NewState[Z_NewState != 0],PossibleStates,Pars)
            States[HighP] = PossibleStates[np.argmax(StateProbSum)]             #Takes the highest value of the probability landscape
            #InsertState = np.sum(PossibleStates*(StateProbSum/np.sum(StateProbSum)))    #Calculates the mean
        
            StateMask = func.attribute2state(F_Selected,Z_Selected,States,Pars)
        for i,x in enumerate(States):    
            Z_NewState = (StateMask == i) * Z_Selected    
            StateProbSum = func.probsum(F_Selected[Z_NewState != 0],Z_NewState[Z_NewState != 0],PossibleStates,Pars)
            States[i] = PossibleStates[np.argmax(StateProbSum)]

    
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
    
    # this plots the FE curve
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)
    fig1.suptitle(Filename, y=1)
    ax1.set_xlabel('Extension [nm]'), ax2.set_xlabel('Free basepairs')
    ax1.set_ylabel('Force [pN]'), ax2.set_ylabel('Probability [AU]')
    ax1.scatter(Z,Force, color="grey",s=1)
    ax1.scatter(Z_Selected,F_Selected, color="blue", s=1)
    ax2.set_xlim([0, Pars['L_bp']+50])    
    ax2.plot(PossibleStates,ProbSum)
    ax2.scatter(PossibleStates[(PeakInd)],Peak)
    #ax2.scatter(Peaks[:,0],Peaks[:,1], color="orange")
    #ax1.set_xlim([-100, Pars['L_bp']/2.8])
    #ax1.set_ylim([-4,25])

    # this plots the Timetrace    
    fig2 = plt.figure()    
    ax3 = fig2.add_subplot(1, 2, 1)
    ax4 = fig2.add_subplot(1, 2, 2, sharey=ax3)
    fig2.suptitle(Filename, y=1)
    ax3.set_xlabel('time [sec]'), ax4.set_xlabel('Probability [AU]')
    ax3.set_ylabel('Extension [bp nm]')
    ax3.set_ylim([0, Pars['L_bp']*Pars['DNAds_nm']+100])
    ax3.scatter(Time,Z, color='grey', s=1)
    ax3.scatter(T_Selected, Z_Selected, color='blue', s=1)
    ax4.plot(ProbSum,PossibleStates*Pars['DNAds_nm'])
    ax4.scatter(Peak,PossibleStates[(PeakInd)]*Pars['DNAds_nm'], color='blue', s=1)
    
    for x in States:
        Ratio = func.ratio(x,Pars)
        Fit = np.array(func.wlc(Force,Pars)*x*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        ax1.plot(Fit,Force, alpha=0.9, linestyle='-.')
        ax3.plot(Time,Fit, alpha=0.9, linestyle='-.')

    #Co-plot the states found initially, to check which states are removed       
    for x in UnMergedStates:
        Ratio = func.ratio(x,Pars)
        Fit = np.array(func.wlc(Force,Pars)*x*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        ax1.plot(Fit,Force, alpha=0.1, linestyle=':')
        ax3.plot(Time,Fit, alpha=0.1, linestyle=':')

   
    fig1.tight_layout()
    #fig1.savefig(Filename[0:-4]+'FoEx_all.png', dpi=800)
    fig1.show()
    fig2.tight_layout()
    #fig2.savefig(Filename[0:-4]+'Time_all.png', dpi=800)    
    fig2.show()
    
    Fignum += 2


#Stepsize,Sigma=func.fit_pdf(steps)
fig3 = plt.figure()
ax5 = fig3.add_subplot(1,1,1)
ax5.hist(steps,  bins = 50, range = [50,250] )
ax5.hist(stacks, bins = 50, range = [50,250])
ax5.set_xlabel('stepsize (bp)')
ax5.set_ylabel('Count')
ax5.set_title("Histogram stepsizes in bp")
ax5.legend(['25 nm steps', 'Stacking transitions'])
#fig3.savefig('hist.png')
fig3.show()
