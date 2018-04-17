# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 16:53:25 2018

@author: nhermans
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

import Functions as func

A = func.fit_2step_gauss(Steps)
print(A)

mu = A[0]
sigma = A[1]
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100) 
plt.plot(x,mlab.normpdf(x, mu, sigma)*A[2]*2)
mu = 2*mu

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100) 

plt.plot(x,mlab.normpdf(x, mu, sigma)*A[3]*2)
Range = [0,400]
Bins = 100
plt.hist(Steps,  bins = Bins, range = Range, lw=0.5, color='blue', label='25 nm stpng', normed=1)
plt.show()