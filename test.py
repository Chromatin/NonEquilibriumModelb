# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:52:49 2018

@author: nhermans
"""
import os  # filenames
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import stats
# from scipy.special import erf
import Functions as func
import Tools
# import peakdetect as pk
from scipy.optimize import curve_fit

T_test=np.array([])                             #array for p values comparing different states
 
P_Cutoff=0.01  
#Calculates the p-value of neighboring states with Welch test
for i,x in enumerate(States):
    if i >0: 
        Prob=stats.ttest_ind((StateMask==i)*Z_Selected,(StateMask==i-1)*Z_Selected, equal_var=False) #get two arrays for t_test
        T_test=np.append(T_test,Prob[1])

#Merges states that are most similar, and are above the p_cutoff minimal significance t-test value
HighP = np.argmax(T_test)
if T_test[HighP] > P_Cutoff:  # Merge the highest p-value states
    DelState = HighP
    if sum((StateMask == HighP + 1) * 1) < sum((StateMask == HighP) * 1): DelState = HighP + 1
    States = np.delete(States, DelState)  # deletes the state in the state array
    StateMask = StateMask - (StateMask > HighP) * 1  # merges the states in the mask
    Z_NewState = (StateMask == HighP) * Z_Selected  # Get all the data for this state to recalculate mean
    MergeStates = True
else:
    MergeStates = False  # Stop merging states
           
#calculate the number of L_unrwap for the new state
if MergeStates:
    #find value for merged state with gaus fit / mean
    StateProbSum = func.probsum(ForceSelected[Z_NewState != 0],Z_NewState[Z_NewState != 0],PossibleStates,Pars)
    InsertState = np.sum(PossibleStates*(StateProbSum/np.sum(StateProbSum)))
    States[HighP] = PossibleStates[np.argmax(StateProbSum)]
    #Ca Pars['L_bp']ulates stepsize
Unwrapsteps=[]
Stacksteps=[]
for x in States:
    if x >= Pars['Fiber0_bp']:
        Unwrapsteps.append(x)
    else:
        Stacksteps.append(x)
Stacksteps=func.state2step(Stacksteps)
Unwrapsteps=func.state2step(Unwrapsteps)
if len(Unwrapsteps)>0: steps.extend(Unwrapsteps)
if len(Stacksteps)>0: stacks.extend(Stacksteps)
#Tools.write_data('AllSteps.txt',Unwrapsteps,Stacksteps)

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
ax2.set_xlim([0, Pars['L_bp']+50])    
ax2.plot(PossibleStates,ProbSum)
ax2.scatter(PossibleStates[(PeakInd)],Peak)
#ax2.scatter(Peaks[:,0],Peaks[:,1], color="orange")
#ax1.set_xlim([-100, Pars['L_bp']/2.8])
#ax1.set_ylim([-4,25])

#plotting
# this plots the Timetrace    
fig2 = plt.figure()    
ax3 = fig2.add_subplot(1, 2, 1)
ax4 = fig2.add_subplot(1, 2, 2, sharey=ax3)
fig2.suptitle(Filename, y=1)
ax3.set_xlabel('time [sec]'), ax4.set_xlabel('Probability [AU]')
ax3.set_ylabel('Extension [bp nm]')
ax3.set_ylim([0, Pars['L_bp']*Pars['DNAds_nm']+100])
ax3.scatter(Time,Z, s=1)
ax4.plot(ProbSum,PossibleStates*Pars['DNAds_nm'])
ax4.scatter(Peak,PossibleStates[(PeakInd)]*Pars['DNAds_nm'], s=1)
ax4.legend(label=States)

for x in States:
    Ratio=func.ratio(x,Pars)
    Fit=np.array(func.wlc(Force,Pars)*x*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'],Fmax_Hook)*Ratio*Pars['ZFiber_nm'])
    ax1.plot(Fit,Force, alpha=0.9, linestyle='-.')
    ax3.plot(Time,Fit, alpha=0.9, linestyle='-.')
    
for x in UnMergedStates:
    Ratio=func.ratio(x,Pars)
    Fit=np.array(func.wlc(Force,Pars)*x*Pars['DNAds_nm'] + func.hook(Force,Pars['k_pN_nm'],Fmax_Hook)*Ratio*Pars['ZFiber_nm'])
    ax1.plot(Fit,Force, alpha=0.1, linestyle=':')
    ax3.plot(Time,Fit, alpha=0.1, linestyle=':')
   

fig1.tight_layout()
#fig1.savefig(Filename[0:-4]+'FoEx_all.png', dpi=800)
fig1.show()
fig2.tight_layout()
#fig2.savefig(Filename[0:-4]+'Time_all.png', dpi=800)    
fig2.show()