'''Checkpoint 2: Part 2: Estimating parameters by maximum likelihood'''

import math as m
import numpy as np
import scipy as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from experiment import Experiment

#Set the number of muons in each experiment
#The true muon lifetime is 2.2 microseconds
num_mu = 1000
tau_true = 2.2

#Use Experiment class to create muons
muon = Experiment(tau_true,num_mu)

#Find the average lifetime
tau_av = muon.average()
print("The average lifetime is: ",tau_av,"microseconds")

#Calculate NLL and minimise
def NLL(tau, muon):
    P = (1/tau)*np.exp(-muon.random/tau)
    nll = - np.sum(np.log(P))
    return nll

#NegLL = NLL(tau_av, muon)
#print("Negative log likelihood is: ",NegLL)
min_NLL = minimize(NLL, tau_av, args=muon)
print("Minimised negative log likelihood is: ",min_NLL["fun"],"microseconds")
print("Minimised tau is: ",min_NLL["x"][0],"microseconds")

#Plot NLL by varying tau around the average tau
param = [1.9,2.4,0.01]
xvals = np.arange(param[0],param[1],param[2])
yvals = np.zeros(len(xvals))
#Create list and append all values below red line
error_list = []
for i in range(len(xvals)):
    yvals[i] = NLL(xvals[i],muon)
    if yvals[i] <= min_NLL["fun"]+0.5:
        error_list.append(xvals[i])
#Plot NLL+0.5 line and find error on tau
#Plot negative log likelihood against tau values in a certain range
line = np.array([min_NLL["fun"]+0.5 for i in range(len(xvals))])
plt.plot(xvals,yvals,xvals,line,'r--')
plt.xlabel("Tau Values")
plt.ylabel("Negative Log Likelihood")
plt.title("Negative Log Likelihood for a range of Tau Values")
plt.show()

#print intercepts of red line and graphs
#print("The line intercepts the graph at tau values: ")
#print(min(error_list), "and" ,max(error_list))
#find error on tau and print
#error is the average of two errors
error_tau = 0.5*((min_NLL["x"][0]-min(error_list))+(max(error_list)-min_NLL["x"][0]))
print("The error on tau is: ",error_tau,"microseconds")

