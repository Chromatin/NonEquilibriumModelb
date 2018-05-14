# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:52:49 2018

@author: nhermans & rrodrigues
"""
import os 
import matplotlib
matplotlib.rcParams['figure.figsize'] = (16, 9)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import Functions as func
import Tools
import time
start_time = time.time()
#import pickle
plt.close('all')                                                                #Close all the figures from previous sessions

folder = r'P:\18S FitFiles\Leiden_wt'
#folder = r'N:\Rick\Fit Files\15x197 H1 Best Traces'

newpath = folder+r'\FiguresPython'                                                   #New path to save the figures
if not os.path.exists(newpath):
    os.makedirs(newpath)

print('Origin:', folder)
print('Destination folder:', newpath)

filenames = os.listdir(folder)
os.chdir(folder)

PlotSelected = True                                                           #Choose to plot selected only
MeasurementERR = 5                                                              #nm

Handles = Tools.Define_Handles(Select=PlotSelected, Pull=True, DelBreaks=True, MinForce=2.5, MaxForce=True, MinZ=0, MaxZ=False, Onepull=True, MedFilt=False)
steps , stacks = [],[]                                                          #used to save data (T-test)
Steps , Stacks = [],[]                                                          #used to save data (Smoothening)
F_Rup_up, Step_up, F_Rup_down, Step_down = [], [], [], []                       #Rupture forces and corresponding jumps
Ruptures = np.empty((0,3)) 

Fignum = 1                                                                      #Used for output line

Filenames = []                                                                  #All .fit files    
for filename in filenames:
    if filename[-4:] == '.fit':
        Filenames.append(filename)

for Filenum, Filename in enumerate(Filenames):

    F, Z, T, Z_Selected = Tools.read_data(Filename)                            #loads the data from the filename
    LogFile = Tools.read_log(Filename[:-4]+'.log')                             #loads the log file with the same name
    if LogFile: Pars = Tools.log_pars(LogFile)                                 #Reads in all the parameters from the logfile
    else: continue                                                                       

    if Pars['FiberStart_bp'] <0: 
        print('<<<<<<<< warning: ',Filename, ': bad fit >>>>>>>>>>>>')
        continue
    print(Filenum+1, "/", len(Filenames), ": ", "%02d" % (int(Pars['N_tot']),), " Nucl. ", Filename, " (Fig. ", Fignum, " & ", Fignum+1, "). Runtime:", np.round(time.time()-start_time, 1), "s", sep='')

    #Remove all datapoints that should not be fitted
    F_Selected, Z_Selected, T_Selected = Tools.handle_data(F, Z, T, Z_Selected, Handles, Pars)

    if len(Z_Selected)<10:  
        print("<<<<<<<<<<<", Filename,'==> No data points left after filtering!>>>>>>>>>>>>')
        continue
    
    PossibleStates, ProbSum, Peak, States, AllStates, Statemask, NewStates, NewStateMask, NewAllStates = func.find_states_prob(F_Selected, Z_Selected, F, Z, Pars, MergeStates=False, Z_Cutoff=2) #Finds States
   
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
    
    # this plots the Force-Extension curve
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 2, 1)
    fig1.suptitle(Filename, y=.99)
    ax1.set_title(r'Extension-Force Curve')
    ax1.set_ylabel(r'Force (pN)')
    ax1.set_xlabel(r'Extension (nm)')
    ax1.scatter(Z, F, color='grey', lw=0.1, s=5)
    ax1.set_ylim([np.min(F_Selected)-0.1*np.max(F_Selected), np.max(F_Selected)+0.1*np.max(F_Selected)])
    ax1.set_xlim([np.min(Z_Selected)-0.1*np.max(Z_Selected), np.max(Z_Selected)+0.1*np.max(Z_Selected)])

    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.set_title(r'Probability Landscape')
    ax2.set_xlabel(r'Free base pair (nm)') #(nm) should be removed
    ax2.set_ylabel(r'Probability (AU)')
    ax2.plot(PossibleStates, ProbSum, label='ProbSum')
    ax2.scatter(States, Peak)

    # this plots the Timetrace
    fig2 = plt.figure()
    fig2.suptitle(Filename, y=.99)

    ax3 = fig2.add_subplot(1, 2, 1)
    ax3.set_title(r'Timetrace Curve')
    ax3.set_xlabel(r'Time (s)')
    ax3.set_ylabel(r'Extension (bp nm)')
    ax3.set_ylim([0, Pars['L_bp']*Pars['DNAds_nm']+100])
    ax3.scatter(T, Z, color='grey', lw=0.1, s=5)

    ax4 = fig2.add_subplot(1, 2, 2, sharey=ax3)
    ax4.set_title(r'Probability Landscape')
    ax4.set_xlabel(r'Probability (AU)')
    ax4.plot(ProbSum,PossibleStates*Pars['DNAds_nm'])
    ax4.scatter(Peak, States*Pars['DNAds_nm'], color='blue')

    ax3.set_xlim([np.min(T_Selected)-0.1*np.max(T_Selected), np.max(T_Selected)+0.1*np.max(T_Selected)])
    ax3.set_ylim([np.min(Z_Selected)-0.1*np.max(Z_Selected), np.max(Z_Selected)+0.1*np.max(Z_Selected)])
    
    if len(States) <1:
        print("<<<<<<<<<<<", Filename,'==> No States were found>>>>>>>>>>>>')
        continue
##############################################################################################
######## Begin Plotting Different States  
    States = NewStates
    Statemask = NewStateMask
    AllStates = NewAllStates    
    
    colors = [plt.cm.brg(each) for each in np.linspace(0, 1, len(States))]     #Color pattern for the states
    dX = 10                                                                     #Offset for text in plot

    #Calculate the rupture forces using a median filter    
    a, b, c, d = func.RuptureForces(F_Selected, Z_Selected, T_Selected, States, Pars, ax1, ax3)
    F_Rup_up.extend(a)
    Step_up.extend(b)
    F_Rup_down.extend(c)    
    Step_down.extend(d)
    
    #Brower-Toland analysis    
    Rups = func.BrowerToland(F_Selected, Z_Selected, T_Selected, States, Pars, ax1, ax3)
    Ruptures = np.append(Ruptures, Rups, axis=0)
    
    Sum = np.sum(Statemask, axis=1)        
    ax1.scatter(Z_Selected[Sum==0], F_Selected[Sum==0], color='black', s=20)    #Datapoint that do not belong to any state
    ax3.scatter(T_Selected[Sum==0], Z_Selected[Sum==0], color='black', s=20)    #Datapoint that do not belong to any state

    #Plot the states and datapoints in the same color
    for j, col in zip(np.arange(len(colors)), colors):
        Mask = Statemask[:,j]
        Fit = AllStates[:,j]
      
        ax1.plot(Fit, F, alpha=0.9, linestyle=':', color=tuple(col)) 
        ax1.scatter(Z_Selected[Mask], F_Selected[Mask], color=tuple(col), s=20, alpha=.6)
#        ax1.text(Fit[np.argmin(np.abs(F-10))], F[np.argmin(np.abs(F-10))], j, horizontalalignment='center')
    
        ax2.vlines(States[j], 0, np.max(Peak), linestyle=':', color=tuple(col))
        ax2.text(States[j], 0, int(States[j]), fontsize=10, horizontalalignment='center', verticalalignment='top', rotation=90)
        
        ax3.plot(T, Fit, alpha=0.9, linestyle=':', color=tuple(col))
        ax3.scatter(T_Selected[Mask], Z_Selected[Mask], color=tuple(col), s=20, alpha=.6)
        
        ax4.hlines(States[j]*Pars['DNAds_nm'], 0, np.max(Peak), color=tuple(col), linestyle=':')
        ax4.text(0, States[j]*Pars['DNAds_nm'], int(States[j]*Pars['DNAds_nm']), fontsize=10, verticalalignment='center', horizontalalignment='right')
                     
    Unwrapsteps = []
    Stacksteps = []
    for x in NewStates:
        if x >= Pars['Fiber0_bp']:
            Unwrapsteps.append(x)
        else:
            Stacksteps.append(x)
    Stacksteps = np.diff(np.array(Stacksteps))
    Unwrapsteps = np.diff(np.array(Unwrapsteps))
    if len(Unwrapsteps)>0: Steps.extend(Unwrapsteps)
    if len(Stacksteps)>0: Stacks.extend(Stacksteps)
######################################################################################################################
    fig1.tight_layout()
#    pickle.dump(fig1, open(newpath+r'\\'+Filename[0:-4]+'_FoEx_all.pickle', 'wb'))            #Saves the figure, so it can be reopend
    fig1.savefig(newpath+r'\\'+Filename[0:-4]+'FoEx_all.png')
    fig1.show()
    
    fig2.tight_layout()
#    pickle.dump(fig2, open(newpath+r'\\'+Filename[0:-4]+'_Time_all.pickle', 'wb'))            #Saves the figure, so it can be reopend
    fig2.savefig(newpath+r'\\'+Filename[0:-4]+'Time_all.png')    
    fig2.show()

    Fignum += 2
#    plt.close('all')


#Brower-Toland Analysis
RFs = Ruptures[:,0]
ln_dFdt_N = np.log(np.divide(Ruptures[:,2],Ruptures[:,1]))
#Remove Ruptures at extensions larger than contour length (ln gets nan value)
RFs = RFs[abs(ln_dFdt_N) < 10e6]
ln_dFdt_N = ln_dFdt_N[abs(ln_dFdt_N) < 10e6]

Fit = np.polyfit(ln_dFdt_N, RFs, 1, full = True)
x = np.linspace(np.min(ln_dFdt_N), np.max(ln_dFdt_N), 10)
a = Fit[0][0]
b = Fit[0][1]
d = Pars['kBT_pN_nm']/a
k_d0 = np.exp(-b/a)/a

fig, ax = plt.subplots()
ax.plot(x, a*x+b, color='red', lw=2, label='Linear Fit')
ax.plot(x, 1.3*x+19, color='green', lw=2, label='Result B-T')
ax.plot(np.log(np.divide(Rups[:,2],Rups[:,1])), Rups[:,0], label='Data', color='red')
ax.scatter(ln_dFdt_N, RFs, label='Data')
ax.set_title("Brower-Toland analysis")
fig.suptitle("d = "+str(np.round(d,2))+"nm, k_D(0)={:.2e}".format(k_d0)+" / sec")
ax.set_xlabel("ln[(dF/dt)/N (pN/s)]")
ax.set_ylabel("Force (pN)")
#ax.set_ylim(5,40)
ax.set_xlim(-4,2)
ax.legend(loc='best', title='Slope:' + str(np.round(a,1)) + '±' + str(np.round(Fit[3][0],1)) + ', intersect:' + str(np.round(b,1)) + '±' + str(np.round(Fit[3][1],1)))
fig.savefig(newpath+r'\\'+'dF_dt_ln.png')

#Plotting a hist of the stepsizes
fig3 = plt.figure()
ax5 = fig3.add_subplot(1,2,1)
ax6 = fig3.add_subplot(1,2,2)
Range = [0,400]
Bins = 50
n = ax5.hist(Steps,  bins=Bins, range=Range, lw=0.5, zorder = 1, color='blue', label='25 nm steps')[0]
ax6.hist(Stacks, bins=int(Bins/2), range=Range, lw=0.5, zorder = 1, color='orange', label='Stacking transitions')

#Fitting double gaussian over 25nm Steps
try: 
    Norm =  Range[-1]/Bins
    D_Gaus = func.fit_2step_gauss(Steps)
    mu = D_Gaus[0]
    sigma = D_Gaus[1]
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100) 
    ax5.plot(x,mlab.normpdf(x, mu, sigma)*D_Gaus[2]*2*Norm, color='red', lw=4, zorder=10, label = 'Gaussian fit')
    mu = 2*mu
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100) 
    ax5.plot(x,mlab.normpdf(x, mu, sigma)*D_Gaus[3]*2*Norm, color='red', lw=4, zorder=10)
    ax5.text(Range[-1]-100, np.max(n)-0.1*np.max(n), 'mean1:'+str(int(D_Gaus[0])), verticalalignment='bottom')
    ax5.text(Range[-1]-100, np.max(n)-0.1*np.max(n), 'mean2:'+str(int(2*D_Gaus[0])), verticalalignment='top')
except:
    print('>>No 25 nm steps to fit gauss')
    
ax5.set_xlabel('stepsize (bp)')
ax5.set_ylabel('Count')
ax5.set_title("Histogram stepsizes 25nm steps")
ax5.legend(loc='best', title='#Samples='+str(len(Filenames))+', Binsize='+str(int(np.max(Range)/Bins))+'bp/bin')
ax6.set_xlabel('stepsize (bp)')
ax6.set_ylabel('Count')
ax6.set_title("Histogram stepsizes stacking steps")
ax6.legend(loc='best', title='#Samples='+str(len(Filenames))+', Binsize='+str(int(np.max(Range)/int(Bins/2)))+'bp/bin')
fig3.tight_layout()
fig3.savefig(newpath+r'\\'+'Hist.png')

#plotting the rupture forces
fig4, ax7 = plt.subplots()
ax7.scatter(F_Rup_up, Step_up, color='red', label='Jump to higher state')           #What should be the errors?
ax7.scatter(F_Rup_down, Step_down, color='Green', label='Jump to lower state')      #What should be the errors?
ax7.set_ylim(0,400)
ax7.set_xlabel('Rupture Forces (pN)')
ax7.set_ylabel('Stepsize (bp)')
ax7.set_title("Rupture forces versus stepsize")
ax7.legend(loc='best')
fig4.savefig(newpath+r'\\'+'RF.png')

print("DONE! Runtime:", np.round(time.time()-start_time, 1), 's',sep='')
