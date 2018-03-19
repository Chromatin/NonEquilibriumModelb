# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:52:49 2018

@author: nhermans
"""
import os 
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
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
    
    Filename = Filename.replace('_', '\_')                                      #Right format for the plot headers

    #Generate FE curves for possible states
    PossibleStates = np.arange(Pars['FiberStart_bp']-200, Pars['L_bp']+50,1)    #range to fit 
    ProbSum = func.probsum(F_Selected, Z_Selected, PossibleStates, Pars)        #Calculate probability landscape
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

    ###########################################################################################################################
    ##############################Trying to define groups of datapoints manually
    UnMergedStates = States       
    
    Z_Dist = np.array([])
    Z_Selected_Sorted = np.sort(Z_Selected)                                     #Sort from low to high Z
    F_Selected_Sorted = np.sort(F_Selected)    
    ZF_Selected = np.vstack((Z_Selected, F_Selected)).T
    
    for i, j in enumerate(Z_Selected_Sorted):                                   #Calc relative Z
            Z_Dist = np.append(Z_Dist, Z_Selected_Sorted[i] - Z_Selected_Sorted[i-1])
    
    Z_DistMask = (abs(Z_Dist) < 6)*1                                            #if dZ > 6 its a 'jump' to a different state
    
    
    fig0 = plt.figure() 
    ax0 = fig0.add_subplot(1,2,1)
    ax00 = fig0.add_subplot(1,2,2, sharex=ax0, sharey=ax0)
    ax0.scatter(Z, Force, color='grey', lw=0.1, s=5, alpha=0.5)    
    ax0.scatter(Z_DistMask*np.max(Z_Selected), F_Selected, s=4, lw=0, color='blue', label=r'D \< 5')
    ax0.scatter(Z_Selected, F_Selected, s=4, lw=0, color='red', label=r'Z_{Selected}')
    ax0.scatter(Z_Selected_Sorted, F_Selected_Sorted, s=4, lw=0, color='orange', label=r'Z_{Selected_{sorted}}')
    l = 0
    NewStates = np.array([])    
    for i,j in enumerate(Z_DistMask):
        if j==0:
            ax0.hlines(F_Selected[i], 0, np.max(Z_Selected), linestyles=':')
            k=i
            if len(Z_Selected[l:k])>=5:                                         #Minimum number of datapoints in a state
                State = States[np.argmin(np.abs(States*Pars['DNAds_nm']-np.average(Z_Selected_Sorted[l:k])))]            
                ax0.vlines(np.average(Z_Selected_Sorted[l:k]), np.min(F_Selected_Sorted[l:k]), np.max(F_Selected_Sorted[l:k]), linestyles=':', color='blue', lw=2)
                ax0.hlines(np.average(F_Selected_Sorted[l:k]), np.min(Z_Selected_Sorted[l:k]), np.max(Z_Selected_Sorted[l:k]), linestyles=':', color='blue', lw=2)              
                NewStates = np.append(NewStates, State)
            l=i
    if len(Z_Selected_Sorted[l:])>=5:
        ax0.vlines(np.average(Z_Selected_Sorted[l:k]), np.min(F_Selected_Sorted[l:]), np.max(F_Selected_Sorted[l:]), linestyles=':', color='blue', lw=2)
        ax0.hlines(np.average(F_Selected_Sorted[l:k]), np.min(Z_Selected_Sorted[l:]), np.max(Z_Selected_Sorted[l:]), linestyles=':', color='blue', lw=2)
        State = States[np.argmin(np.abs(States*Pars['DNAds_nm']-np.average(Z_Selected_Sorted[l:])))]
        NewStates = np.append(NewStates, State)        
    
    for x in NewStates:
        Ratio = func.ratio(x,Pars)
        Fit = np.array(func.wlc(Force,Pars)*x*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        ax0.plot(Fit,Force, alpha=0.9, linestyle='-.')
    
    for x in UnMergedStates:
        Ratio = func.ratio(x,Pars)
        Fit = np.array(func.wlc(Force,Pars)*x*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        ax0.plot(Fit,Force, alpha=0.1, linestyle=':')
            

    ##############################Using DBSCAN to find groups of datapoints
    from sklearn.cluster import DBSCAN

    # Compute DBSCAN
    db = DBSCAN(eps=10, min_samples=7).fit(ZF_Selected)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    unique_labels = set(labels)

    Av = np.empty(shape=[0, 2])

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
           col = [0, 0, 0, 1] # Black used for noise.  
    
        class_member_mask = (labels == k)
    
        xy = ZF_Selected[class_member_mask & core_samples_mask]
        ax00.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=7)
        
        Av = np.append(Av, [np.array([np.average(xy[:,0]),np.average(xy[:, 1])])], axis=0)

        xy = ZF_Selected[class_member_mask & ~core_samples_mask]
        ax00.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=4)
    
 
    Av = np.delete(Av, -1, 0)                                                   #Remove NAN in last row

    ax00.scatter(Av[:,0], Av[:,1], s=500, marker='x', zorder=1)    
    ax00.scatter(Z,Force, color='grey', lw=0.1, s=5, alpha=0.5)    
    
    for x in States:
        Ratio = func.ratio(x,Pars)
        Fit = np.array(func.wlc(Force,Pars)*x*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        ax00.plot(Fit,Force, alpha=0.9, linestyle=':')
    
    
    ax00.set_title(r'Clusters Bound by DBscan')
    ax00.set_ylabel(r'\textbf{Force} (pN)')
    ax00.set_xlabel(r"\textbf{Extension} (nm)")     
    
    ax0.legend()
    ax0.set_ylabel(r'\textbf{Force} (pN)')
    ax0.set_xlabel(r"\textbf{Extension} (nm)")    
    ax0.set_title('Clusters Found Manually')
    ax0.set_ylim(0,np.max(F_Selected)+.1*np.max(F_Selected))
    ax0.set_xlim(0,np.max(Z_Selected)+.1*np.max(Z_Selected))
    
    fig0.suptitle(Filename, y=.99)
    fig0.show()  
    
    ###########################################################################################################################
  
    
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
    ax1.scatter(Z_Selected,F_Selected, color="blue", s=1)   
    ax2.plot(0.34*PossibleStates,ProbSum)                                       #*0.34 should be removed
    ax2.scatter(0.34*PossibleStates[(PeakInd)],Peak)                            #*0.34 should be removed
    ax1.set_xlim([np.min(Z)-0.1*np.max(Z), np.max(Z)+0.1*np.max(Z)])
    ax1.set_ylim([np.min(Force)-0.1*np.max(Force), np.max(Force)+0.1*np.max(Force)])
    #ax2.set_xlim([np.min(PossibleStates)-0.1*np.max(PossibleStates), np.max(PossibleStates)+0.1*np.max(PossibleStates)])
    

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
    
    Fignum += 3


#Stepsize,Sigma=func.fit_pdf(steps)
fig3 = plt.figure()
ax5 = fig3.add_subplot(1,1,1)
ax5.hist(steps,  bins = 50, range = [50,250], label='25 nm steps')
ax5.hist(stacks, bins = 50, range = [50,250], label='Stacking transitions')
ax5.set_xlabel('stepsize (bp)')
ax5.set_ylabel('Count')
ax5.set_title("Histogram stepsizes in bp")
ax5.legend()
#fig3.savefig('hist.png')
fig3.show()
