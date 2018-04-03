# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:52:49 2018

@author: nhermans & rrodrigues
"""
import os 
import matplotlib
matplotlib.rcParams['figure.figsize'] = (15, 10)

#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import numpy as np
import Functions as func
import Tools
import pickle

plt.close('all')                                                                #Close all the figures from previous sessions

folder = r'N:\Rick\Tweezer data\Pythontestfit\ProbabilityTest'
folder = folder.replace('\\', '\\\\')                                           #Replaces \ for \\

filenames = os.listdir(folder)
os.chdir(folder)

PlotSelected = True                                                             #Choose to plot selected only

Handles = Tools.Define_Handles(Select=PlotSelected)
steps , stacks = [],[]                                                          #used to save data (T-test)
Steps , Stacks = [],[]                                                          #used to save data (Smoothening)
F_rup, dZ_rup = np.array([]), np.array([])                                      #Rupture forces and corresponding jumps

Fignum, Progress = 1, 1

Number = 0                                                                      #Total number of loops
for filename in filenames:
    if filename[-4:] == '.fit':
        Number += 1
        
for Filenum, Filename in enumerate(filenames):
    if Filename[-4:] != '.fit' :
        continue
    Force, Time, Z, Z_Selected = Tools.read_data(Filename)                      #loads the data from the filename
    LogFile = Tools.read_log(Filename[:-4]+'.log')                              #loads the log file with the same name
    Pars = Tools.log_pars(LogFile)                                              #Reads in all the parameters from the logfile

    if Pars['FiberStart_bp'] <0: 
        print('<<<<<<<< warning: ',Filename, ': bad fit >>>>>>>>>>>>')
        continue
    print(Progress, "/", Number, ":", int(Pars['N_tot']), "Nucleosomes in", Filename, "( Fig.", Fignum, "&", Fignum+1, ").")
    Progress += 1
    #Remove all datapoints that should not be fitted
    Z_Selected, F_Selected, T_Selected = Tools.handle_data(Force, Z, Time, Z_Selected, Handles, Pars)

    if len(Z_Selected)<10:  
        print("<<<<<<<<<<<", Filename,'==> No data points left after filtering!>>>>>>>>>>>>')
        continue
    
    PossibleStates = np.arange(Pars['FiberStart_bp']-200, Pars['L_bp']+50,1)
    ProbSum = func.probsum(F_Selected, Z_Selected, PossibleStates, Pars)
    ProbProd, Pz, N, pz = func.probprod(F_Selected, Z_Selected, PossibleStates, Pars)      #Product of Probabilities, #workinprogress
    PeakInd, Peak = func.findpeaks(ProbSum, 25)                                 #Find Peaks
    PeakIndProd, PeakProd = func.findpeaks(ProbProd, 25)                        #Find Peaks        
    Starting_States = PossibleStates[PeakInd]                                   #Defines state for each peak
    States = func.find_states_prob(F_Selected,Z_Selected,Pars, MergeStates=False, P_Cutoff=0.1) #Finds States
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
    fig1.suptitle(Filename, y=.99)
    ax1.set_title(r'Extension-Force Curve')
    ax1.set_ylabel(r'Force (pN)')
    ax1.set_xlabel(r'Extension (nm)')
    ax1.scatter(Z, Force, color='grey', lw=0.1, s=5)
    if PlotSelected:
        ax1.set_ylim([np.min(F_Selected)-0.1*np.max(F_Selected), np.max(F_Selected)+0.1*np.max(F_Selected)])
        ax1.set_xlim([np.min(Z_Selected)-0.1*np.max(Z_Selected), np.max(Z_Selected)+0.1*np.max(Z_Selected)])

    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.set_title(r'Probability Landscape')
    ax2.set_xlabel(r'Free base pair (nm)') #(nm) should be removed
    ax2.set_ylabel(r'Probability (AU)')
    ax2.plot(PossibleStates,ProbSum, alpha=0.1, label='ProbSum')
#    ax2.plot(PossibleStates,ProbProd, color='orange', label='ProbProd')         #Product of Probabilities, #workinprogress
    ax2.scatter(PossibleStates[(PeakInd)],Peak, alpha=0.1)
#    ax2.scatter(PossibleStates[(PeakIndProd)],PeakProd, color='orange')
    ax2.set_ylim(0,np.max(ProbSum)) #Just here because ProbProd is extremely large

    # this plots the Timetrace
    fig2 = plt.figure()
    fig2.suptitle(Filename, y=.99)

    ax3 = fig2.add_subplot(1, 2, 1)
    ax3.set_title(r'Timetrace Curve')
    ax3.set_xlabel(r'Time (s)')
    ax3.set_ylabel(r'Extension (bp nm)')
    ax3.set_ylim([0, Pars['L_bp']*Pars['DNAds_nm']+100])
    ax3.scatter(Time,Z, color='grey', lw=0.1, s=5)
    if PlotSelected:
        ax3.set_xlim([np.min(T_Selected)-0.1*np.max(T_Selected), np.max(T_Selected)+0.1*np.max(T_Selected)])
        ax3.set_ylim([np.min(Z_Selected)-0.1*np.max(Z_Selected), np.max(Z_Selected)+0.1*np.max(Z_Selected)])

    ax4 = fig2.add_subplot(1, 2, 2, sharey=ax3)
    ax4.set_title(r'Probability Landscape')
    ax4.set_xlabel(r'Probability (AU)')
    ax4.plot(ProbSum,PossibleStates*Pars['DNAds_nm'], alpha=0.1)
    ax4.scatter(Peak,PossibleStates[(PeakInd)]*Pars['DNAds_nm'], color='blue', alpha=0.1)
    
    #Plot the states found
    for x in States:
        Ratio = func.ratio(x,Pars)
        Fit = np.array(func.wlc(Force,Pars)*x*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        ax1.plot(Fit, Force, alpha=0.1, linestyle='-.')
        ax3.plot(Time,Fit, alpha=0.1, linestyle='-.')

##############################################################################################
######## Begin Smoothening
#    Smoothness = 30
#    SmoothProbProd = func.Conv(ProbProd,Smoothness)
#    SmoothPeakIndProd, SmoothPeakProd = func.findpeaks(SmoothProbProd, 30)
#    SmoothStatesProd = PossibleStates[SmoothPeakIndProd]  
#    
#    ax2.plot(PossibleStates, SmoothProbProd, color='purple', label='SmoothProdProd')
#    ax2.scatter(PossibleStates[(SmoothPeakIndProd)], SmoothPeakProd, color='purple')
   
    Smoothness = 70
    SmoothProbSum = func.Conv(ProbSum,Smoothness)
    SmoothPeakInd, SmoothPeak = func.findpeaks(SmoothProbSum, 25)
    SmoothStates = PossibleStates[SmoothPeakInd]

    ax2.plot(PossibleStates, SmoothProbSum, 'g-', lw=2, label='SmoothProbSum')
    ax2.scatter(PossibleStates[(SmoothPeakInd)],SmoothPeak, color='green')

    
    ax4.plot(SmoothProbSum, PossibleStates*Pars['DNAds_nm'], color='green')
    ax4.scatter(SmoothPeak,PossibleStates[(SmoothPeakInd)]*Pars['DNAds_nm'], color='green')
    
    Statemask = func.attribute2state(F_Selected,Z_Selected,SmoothStates, Pars)  #For each datapoint to which state it belongs
    AllStates = np.empty(shape=[len(Force),2,len(SmoothStates)])                #3d array of the states
    
    #Making a 3d array containing all states afther smoothening: Fit==AllStates[:,0,i], Force==AllStates[:,1,i] for state i
    for i, x in enumerate(SmoothStates):
        Ratio = func.ratio(x,Pars)
        Fit = np.array(func.wlc(Force,Pars)*x*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        ForceFit = np.vstack((Fit, Force)).T
        AllStates[:,:,i] = ForceFit        

    colors = [plt.cm.brg(each) for each in np.linspace(0, 1, len(SmoothStates))]#Color pattern for the states
    dX = 10                                                                     #Offset for text in plot
    
    #Plot the states and datapoints in the same color
    for j, col in zip(np.arange(len(colors)), colors):
        Mask = Statemask==j
        Fit = AllStates[:,0,j]
        Force = AllStates[:,1,j]
        
        ax1.plot(Fit, Force, alpha=0.9, linestyle=':', color=tuple(col)) 
        ax1.scatter(Z_Selected[Mask], F_Selected[Mask], color=tuple(col), s=5)
        
        ax2.vlines(SmoothStates[j], 0, SmoothPeak[j], linestyle=':', color=tuple(col))
        ax2.text(SmoothStates[j], SmoothPeak[j]+dX, int(SmoothStates[j]), fontsize=8, horizontalalignment='center')
        
        ax3.plot(Time, Fit, alpha=0.9, linestyle=':', color=tuple(col))
        ax3.scatter(T_Selected[Mask], Z_Selected[Mask], color=tuple(col), s=5)
        
        ax4.hlines(SmoothStates[j]*Pars['DNAds_nm'], 0, SmoothPeak[j], color=tuple(col), linestyle=':')
        ax4.text(SmoothPeak[j]+dX, SmoothStates[j]*Pars['DNAds_nm'], int(SmoothStates[j]*Pars['DNAds_nm']), fontsize=8, verticalalignment='center')
    
        #Rupture forces
        if j < len(SmoothStates)-1:
            Ruptureforce = np.mean((F_Selected[Mask])[-3:-1])                               #The 3 last datapoint in a group
            start = Fit[np.argmin(np.abs(Force-Ruptureforce))]
            stop = (AllStates[:,0,j+1])[np.argmin(np.abs(AllStates[:,1,j+1]-Ruptureforce))] #Same as start, but then for the next state
            ax1.hlines(Ruptureforce, start, stop, color='black')
            F_rup = np.append(F_rup, Ruptureforce)
            dZ_rup = np.append(dZ_rup, stop-start)
      
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

######################################################################################################################
    ax2.legend(loc='best')
    fig1.tight_layout()
    pickle.dump(fig1, open(Filename[0:-4]+'.FoEx_all.pickle', 'wb'))            #Saves the figure, so it can be reopend
    fig1.savefig(Filename[0:-4]+'FoEx_all.pdf')
    fig1.show()
    
    fig2.tight_layout()
    pickle.dump(fig2, open(Filename[0:-4]+'.Time_all.pickle', 'wb'))            #Saves the figure, so it can be reopend
    fig2.savefig(Filename[0:-4]+'Time_all.pdf')    
    fig2.show()

    Fignum += 2

#Plotting a hist of the stepsizes
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
fig3.savefig('Hist.pdf', format='pdf')

#plotting the rupture forces
fig4, ax7 = plt.subplots()
ax7.scatter(F_rup, dZ_rup, color='blue')       #What should be the errors?
ax7.set_xlabel('Rupture Forces (pN)')
ax7.set_ylabel('Jump in Z (nm)')
ax7.set_title("Rupture forces versus jump in z")
fig4.savefig('RF.pdf', format='pdf')
