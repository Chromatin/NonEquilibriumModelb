# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:52:49 2018

@author: nhermans & rrodrigues
"""
import os 
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import numpy as np
import Functions as func
import Tools
import pickle
from scipy import stats
from sklearn.cluster import DBSCAN

folder = 'N:\\Rick\\Tweezer data\\Pythontestfit'
folder = 'P:\\NonEqData\\H1_197\\Best Traces'

filenames = os.listdir(folder)
os.chdir(folder)

Handles = Tools.Define_Handles()
steps , stacks = [],[]                                                          #used to save data (T-test)
Steps , Stacks = [],[]                                                          #used to save data (Clustering)
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

    #Generate FE curves for possible states
    PossibleStates = np.arange(Pars['FiberStart_bp']-200, Pars['L_bp']+50,1)    #range to fit 
    ProbSum = func.probsum(F_Selected, Z_Selected, PossibleStates, Pars)        #Calculate probability landscape
    PeakInd, Peak = func.findpeaks(ProbSum, 25)                                 #Find Peaks    
    States = PossibleStates[PeakInd]                                            #Defines state for each peak

    #Calculate for each datapoint which state it most likely belongs too 
    StateMask = func.attribute2state(F_Selected,Z_Selected,States,Pars)
    
    #Remove states with 5 or less datapoints
    RemoveStates = func.removestates(StateMask)
    if len(RemoveStates)>0:
        States = np.delete(States, RemoveStates)
        StateMask = func.attribute2state(F_Selected, Z_Selected, States, Pars)

    MergeStates=True
    P_Cutoff=0.1
    
    T_test=np.array([])
    for i, x in enumerate(States):
        if i > 0:
            Prob = stats.ttest_ind((StateMask == i) * Z_Selected, (StateMask == i - 1) * Z_Selected,equal_var=False)  # get two arrays for t_test
            T_test = np.append(T_test, Prob[1])

    while MergeStates == True:  # remove states untill all states are significantly different
    
        # Merges states that are most similar, and are above the p_cutoff minimal significance t-test value
        HighP = np.argmax(T_test)
        if T_test[HighP] > P_Cutoff:  # Merge the highest p-value states
            DelState, MergedState = HighP, HighP+1
            if sum((StateMask == HighP + 1) * 1) < sum((StateMask == HighP) * 1): 
                DelState = HighP + 1
                MergedState = HighP
            #Recalculate th     
            Prob = stats.ttest_ind((StateMask == MergedState ) * Z_Selected, (StateMask == HighP - 1) * Z_Selected,equal_var=False)  # get two arrays for t_test
            T_test[HighP-1] = Prob[1]
            Prob = stats.ttest_ind((StateMask == MergedState ) * Z_Selected, (StateMask == HighP - 1) * Z_Selected,equal_var=False)  # get two arrays for t_test
            T_test[HighP+1] = Prob[1]
            T_test=np.delete(T_test,HighP)
            States = np.delete(States, DelState)  # deletes the state in the state array
            StateMask = StateMask - (StateMask == HighP + 1) * 1  # merges the states in the mask
            Z_NewState = (StateMask == HighP) * Z_Selected  # Get all the data for this state to recalculate mean
            MergeStates = True
        else:
            MergeStates = False  # Stop merging states
                   
        #calculate the number of L_unrwap for the new state
        if MergeStates:
            #find value for merged state with gaus fit / mean
            StateProbSum = func.probsum(F_Selected[Z_NewState != 0],Z_NewState[Z_NewState != 0],PossibleStates,Pars)
            States[HighP] = PossibleStates[np.argmax(StateProbSum)]                     #Takes the highest value of the probability landscape

    ###########################################################################################################################
    #Finding groups/clusters of datapoints      
    ZF_Selected = np.vstack((Z_Selected, F_Selected)).T

    #Force-Extension plot
    fig0 = plt.figure()
    fig0.suptitle(Filename, y=.99)
    ax0 = fig0.add_subplot(1,1,1) 
    ax0.scatter(Z,Force, color='grey', lw=0.1, s=5, alpha=0.5)

    # Compute DBSCAN
    db = DBSCAN(eps=10, min_samples=7).fit(ZF_Selected)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)

    NewStates = np.array([])
    AllStates = np.empty(shape=[len(Force),2,len(PossibleStates)])
    Av = np.empty(shape=[0, 2])

    #Plotting the different clusters in different colors, also calculating the mean of each cluster
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
           col = [0, 0, 0, 1] # Black used for noise.  

        class_member_mask = (labels == k)

        xy = ZF_Selected[class_member_mask & core_samples_mask]
        ax0.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=3)
        
        if k != -1:
            Av = np.append(Av, [np.array([np.mean(xy[:,0]),np.mean(xy[:, 1])])], axis=0)

        xy = ZF_Selected[class_member_mask & ~core_samples_mask]
        ax0.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=1) 
    
    #Making a 3d array containing all the possible states
    for i, x in enumerate(PossibleStates):
        Ratio = func.ratio(x,Pars)
        Fit = np.array(func.wlc(Force,Pars)*x*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        ForceFit = np.vstack((Fit, Force)).T
        AllStates[:,:,i] = ForceFit        
        #print(filenames[Filenum], "(#", Fignum, ")is at", "{0:0.1f}".format(i/len(PossibleStates)*100),"%") #Just to see if the programm is actually running 
                                                                                                            #and how long it will take
    #Calculating the states that are closest to the means computed above
    for I,J in enumerate(Av[:,0]):   
        AllDist = np.array([])
        for i,j in enumerate(AllStates[0,0,:]):
            Dist = np.abs(np.subtract(AllStates[:,:,i], Av[I,:]))
            Dist = np.square(Dist)
            Dist = np.sum(Dist, axis=1)
            AllDist = np.append(AllDist, np.min(Dist))
        NewStates = np.append(NewStates, PossibleStates[np.argmin(AllDist)])

    #Plotting the corresponding states in the same color as the clusters
    for i, col in zip(enumerate(NewStates), colors):
        Ratio = func.ratio(i[1],Pars)
        Fit = np.array(func.wlc(Force,Pars)*i[1]*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        ax0.plot(Fit,Force, alpha=0.9, linestyle=':', color=tuple(col))
    ax0.set_title(r'Force-Extension Curve of Chromatin Fibre')
    ax0.set_ylabel(r'\textbf{Force} (pN)')
    ax0.set_xlabel(r"\textbf{Extension} (nm)")     
#    ax0.set_ylim(0,np.max(F_Selected)+.1*np.max(F_Selected))
#    ax0.set_xlim(0,np.max(Z_Selected)+.1*np.max(Z_Selected))

    #Timetrace Plot
    fig00 = plt.figure()
    ax00 = fig00.add_subplot(1, 1, 1)
    fig00.suptitle(Filename, y=.99)
    ax0.set_title(" ")
    ax00.set_xlabel(r'\textbf{Time} (s)')
    ax00.set_ylabel(r'\textbf{Extension} (bp nm)')
    ax00.set_ylim([0, Pars['L_bp']*Pars['DNAds_nm']+100])
    ax00.scatter(Time,Z,  c=Time, cmap='gray', lw=0.1, s=5)
    ax00.scatter(T_Selected, Z_Selected, color='blue', s=1)
    ax00.set_xlim([np.min(Time)-0.1*np.max(Time), np.max(Time)+0.1*np.max(Time)])
    ax00.set_ylim([np.min(Z)-0.1*np.max(Z), np.max(Z)+0.1*np.max(Z)])

    for i, col in zip(enumerate(NewStates), colors):
        Ratio = func.ratio(i[1],Pars)
        Fit = np.array(func.wlc(Force,Pars)*i[1]*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        ax00.plot(Time,Fit, alpha=0.9, linestyle='-.', color=tuple(col))

    Filename = Filename.replace( '\_', '_')                                     #Right format to safe the figure

    pickle.dump(fig0, open(Filename[0:-4]+'.FoEx_all.pickle', 'wb'))            #Saves the figure, so it can be reopend
    pickle.dump(fig00, open(Filename[0:-4]+'.Time_all.pickle', 'wb'))
    
    #fig0.savefig(Filename[0:-4]+'FoEx_all.pdf', format='pdf')
    fig0.show()     
    #fig00.savefig(Filename[0:-4]+'Time_all.pdf', format='pdf')
    fig00.show()
    
    #Calculates stepsize
    Unwrapsteps = []
    Stacksteps = []
    for x in NewStates:
        if x >= Pars['Fiber0_bp']:
            Unwrapsteps.append(x)
        else:
            Stacksteps.append(x)
    Stacksteps = np.diff(np.array(Stacksteps)) #func.state2step(Stacksteps)
    Unwrapsteps = np.diff(np.array(Unwrapsteps)) #func.state2step(Unwrapsteps)
    if len(Unwrapsteps)>0: Steps.extend(Unwrapsteps)
    if len(Stacksteps)>0: Stacks.extend(Stacksteps)
    ###########################################################################################################################
    
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
    ax2.set_xlim([np.min(PossibleStates)-0.1*np.max(PossibleStates), np.max(PossibleStates)+0.1*np.max(PossibleStates)])

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

    fig1.tight_layout()
    #fig1.savefig(Filename[0:-4]+'FoEx_all.pdf')
    fig1.show()
    fig2.tight_layout()
    #fig2.savefig(Filename[0:-4]+'Time_all.pdf')    
    fig2.show()

    Fignum += 1


#Stepsize,Sigma=func.fit_pdf(steps)
fig3 = plt.figure()
ax5 = fig3.add_subplot(1,2,1)
ax6 = fig3.add_subplot(1,2,2)
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
ax6.set_title("Histogram stepsizes in bp using clustering")
ax6.legend()
fig3.tight_layout()
#fig3.savefig('hist.png')
