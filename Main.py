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

folder = r'N:\Rick\Fit Files\Pythontestfit'
folder = folder.replace('\\', '\\\\')                                           #Replaces \ for \\

newpath = folder+r'\\Figures'                                                   #New path to save the figures
if not os.path.exists(newpath):
    os.makedirs(newpath)

print('Origin:', folder)
print('Destination folder:', newpath)

filenames = os.listdir(folder)
os.chdir(folder)

PlotSelected = True                                                             #Choose to plot selected only

MeasurementERR = 5 #nm

Handles = Tools.Define_Handles(Select=PlotSelected)
steps , stacks = [],[]                                                          #used to save data (T-test)
Steps , Stacks = [],[]                                                          #used to save data (Smoothening)
F_rup, dZ_rup = np.array([]), np.array([])                                      #Rupture forces and corresponding jumps

Fignum, Progress = 1, 1                                                         #Used for output line

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
    
    PossibleStates, ProbSum, Peak, PeakInd, States, LocalStiffness = func.find_states_prob(F_Selected,Z_Selected,Pars, MergeStates=False, P_Cutoff=0.1) #Finds States
 
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
    ax2.plot(PossibleStates,ProbSum, label='ProbSum')
    ax2.scatter(PossibleStates[(PeakInd)],Peak)

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
    ax4.plot(ProbSum,PossibleStates*Pars['DNAds_nm'])
    ax4.scatter(Peak,PossibleStates[(PeakInd)]*Pars['DNAds_nm'], color='blue')
    
    #Plot the states found initially
    for x in States:
        Ratio = func.ratio(x,Pars)
        Fit = np.array(func.wlc(Force,Pars)*x*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        ax1.plot(Fit, Force, alpha=0.1, linestyle='-.')
        ax3.plot(Time,Fit, alpha=0.1, linestyle='-.')

##############################################################################################
######## Begin Plotting Different States

    #Remove states with X or less datapoints, not nessesary at this moment ==> This is done in func.find_states_prob   
#    States, Peak, Statemask = func.MinNumOfPoints(States, Peak, Statemask, F_Selected, Z_Selected, Pars, X=2)

    #Making a 2d array containing all states: Fit==AllStates[:,i]
    AllStates = np.empty(shape=[len(Z), len(States)])                           #2d array of the states  
    AllStates_Selected = np.empty(shape=[len(Z_Selected), len(States)])                           #2d array of the states  \  
    for i, x in enumerate(States):
        Ratio = func.ratio(x,Pars)
        Fit = np.array(func.wlc(Force,Pars)*x*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        Fit_Selected = np.array(func.wlc(F_Selected,Pars)*x*Pars['DNAds_nm'] + func.hook(F_Selected,Pars['k_pN_nm'])*Ratio*Pars['ZFiber_nm'])
        AllStates[:,i] = Fit        
        AllStates_Selected[:,i] = Fit_Selected        
    
    
    sigma = np.sqrt(Pars['kBT_pN_nm']/LocalStiffness)    
    std = np.sqrt(MeasurementERR**2 + np.multiply(sigma,sigma))                         #sqrt([measuring error]^2 + [thermal fluctuations]^2) 
    Z_Score = (func.z_score(Z_Selected, AllStates_Selected, std))        
     
    StateMaskZscore = np.abs(Z_Score) < 3

        
    Statemask = func.attribute2state(F_Selected, Z_Selected, States, Pars)      #For each datapoint to which state it belongs

    colors = [plt.cm.Set1(each) for each in np.linspace(0, 1, len(States))]      #Color pattern for the states
    dX = 10                                                                     #Offset for text in plot

    #Calculate the rupture forces using a median filter    
#    func.RuptureForces(Z_Selected, F_Selected, States, Pars, ax1)
    Sum = np.sum(StateMaskZscore, axis=1)        
    ax1.scatter(Z_Selected[Sum==0], F_Selected[Sum==0], color='black', s=30)    #Datapoint that do not belong to any state

    #Plot the states and datapoints in the same color
    for j, col in zip(np.arange(len(colors)), colors):
        Mask = Statemask==j
        Fit = AllStates[:,j]
      
#        ax1.plot(Fit, Force, alpha=0.9, linestyle=':', color=tuple(col)) 
#        ax1.scatter(Z_Selected[Mask], F_Selected[Mask], color=tuple(col), s=5)


        ax1.plot(Fit, Force, alpha=0.9, linestyle=':', color=tuple(col)) 
        ax1.scatter(Z_Selected[StateMaskZscore[:,j]], F_Selected[StateMaskZscore[:,j]], color=tuple(col), s=42, alpha=.6)

        ax2.vlines(States[j], 0, Peak[j], linestyle=':', color=tuple(col))
        ax2.text(States[j], Peak[j]+dX, int(States[j]), fontsize=8, horizontalalignment='center')
        
        ax3.plot(Time, Fit, alpha=0.9, linestyle=':', color=tuple(col))
        ax3.scatter(T_Selected[Mask], Z_Selected[Mask], color=tuple(col), s=5)
        
        ax4.hlines(States[j]*Pars['DNAds_nm'], 0, Peak[j], color=tuple(col), linestyle=':')
        ax4.text(Peak[j]+dX, States[j]*Pars['DNAds_nm'], int(States[j]*Pars['DNAds_nm']), fontsize=8, verticalalignment='center')
        
        
        #Rupture forces
        if j < len(States)-1:   #This should be done by median filter & in basepairs
            Ruptureforce = np.mean((F_Selected[Mask])[-4:-1])                               #The 4 last datapoint in a group
            start = Fit[np.argmin(np.abs(Force-Ruptureforce))]
            stop = (AllStates[:,j+1])[np.argmin(Force-Ruptureforce)] #Same as start, but then for the next state
#            ax1.hlines(Ruptureforce, start, stop, color='black')
            F_rup = np.append(F_rup, Ruptureforce)
            dZ_rup = np.append(dZ_rup, stop-start)
      
    Unwrapsteps = []
    Stacksteps = []
    for x in States:
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
#    pickle.dump(fig1, open(newpath+r'\\'+Filename[0:-4]+'_FoEx_all.pickle', 'wb'))            #Saves the figure, so it can be reopend
    fig1.savefig(newpath+r'\\'+Filename[0:-4]+'FoEx_all.pdf', format='pdf')
    fig1.show()
    
    fig2.tight_layout()
#    pickle.dump(fig2, open(newpath+r'\\'+Filename[0:-4]+'_Time_all.pickle', 'wb'))            #Saves the figure, so it can be reopend
    fig2.savefig(newpath+r'\\'+Filename[0:-4]+'Time_all.pdf', format='pdf')    
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
ax5.set_title("Histogram stepsizes in bp before T-test")
ax5.legend(loc='best')
ax6.set_xlabel('stepsize (bp)')
ax6.set_ylabel('Count')
ax6.set_title("Histogram stepsizes in bp after T-test")
ax6.legend(loc='best')
fig3.tight_layout()
fig3.savefig(newpath+r'\\'+'Hist.pdf', format='pdf')

#plotting the rupture forces
fig4, ax7 = plt.subplots()
ax7.scatter(F_rup, dZ_rup, color='blue')       #What should be the errors?
ax7.set_xlabel('Rupture Forces (pN)')
ax7.set_ylabel('Jump in Z (nm)')
ax7.set_title("Rupture forces versus jump in z")
fig4.savefig(newpath+r'\\'+'RF.pdf', format='pdf')
