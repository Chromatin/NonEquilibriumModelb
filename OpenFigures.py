import matplotlib.pyplot as plt
import pickle
import os 
from tkinter import filedialog
from tkinter import *

plt.close('all')                                                                #Close all the figures from previous sessions

root = Tk()
root.withdraw()                                                   

folder =  'N:\\Rick\\Tweezer data\\Pythontestfit\\New folder'#filedialog.askdirectory() #folder with chromosome sequence files (note, do not put other files in this folder)
filenames = os.listdir(folder)
os.chdir(folder)

for Filename in filenames:
    if Filename[-7:] == '.pickle' :
        figx = pickle.load(open(Filename, 'rb'))
        figx.show()
