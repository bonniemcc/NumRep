'''Project Part 2: 
Import time and theta values from datafile.txt.
Perform a 2-dimensional maximum likelihood fit to determine the
best fit values of the parameters (F, tau1, tau2) and their errors.
Find errors by varying one parameter whilst keeping all others fixed.
Present the results in a table.
'''

import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def PDF(time,theta,x0,N):
    F = x0[0]
    tau1 = x0[1]
    tau2 = x0[2]
    N1 = N[0]
    N2 = N[1]
    return F*(1/N1)*(1+np.cos(theta)*np.cos(theta))*np.exp(-time/tau1) + (1-F)*(1/N2)*3*np.sin(theta)*np.sin(theta)*np.exp(-time/tau2)

def mymin(fun,x0,time, theta):
    bnds = ((0,1.0),(0,np.inf),(0,np.inf))
    return minimize(fun,x0,args=(time, theta),bounds=bnds)

def NLL(PDF):
    return - np.sum(np.log(PDF))

def minNLL(fun,x0,time,theta):
    bnds = ((0,1.0),(0,np.inf),(0,np.inf))
    #print("NLL for PDF: ",NLL(P))
    #minimise nll
    #print("Minimised NLL for PDF is: ",mymin(NLL,x0)["fun"])
    #minimise parameters
    print("Minimised F is: ",minimize(fun,x0,args=(time,theta), bounds=bnds)["x"][0])
    #print("Minimised F is: ",mymin(NLL,x0,args=(time,theta))["x"][0])
    #print("Minimised tau1 is: ",mymin(NLL,x0,args=(time,theta))["x"][1])
    #print("Minimised tau2 is: ",mymin(NLL,x0,args=(time,theta))["x"][2])

    
def main():
    
    #get data from text file
    data = np.loadtxt(open('datafile-Xdecay.txt', 'r'))
    time,theta = data[:,0],data[:,1]
    #set parameter values
    x0 = [0.5,1,2]
    N = [1,1]
    
    #bnds = ((0,1.0),(0,np.inf),(0,np.inf))
    #result = minimize(NLL, x0, args=(time, theta), bounds=bnds)
    #print(result)
    
    #find nll using only time information, set theta as pi
    theta_const = np.full(len(time),m.pi)
    '''P1 = np.zeros(len(time))
    for i in range(len(time)):
        P1[i] = PDF(time[i],theta_const[i],x0,N)
    minNLL(P1,x0,NLL,mymin)'''
    minNLL(NLL,x0,time,theta_const)

    #find nll using only theta information, set time as 0
    time_const = np.zeros(len(theta))
    '''P2 = np.zeros(len(theta))
    for i in range(len(theta)):
        P2[i] = PDF(time_const[i],theta[i],x0,N)
    minNLL(P2,x0,NLL,mymin)'''
    minNLL(PDF,x0,NLL,mymin,args=(time_const,theta))

    #print(mymin(NLL,x0))
    #find errors

    #find nll using both time and theta information
    #P3 = np.zeros(len(theta))
    #for i in range(len(theta)):
    #P3[i] = PDF(time[i],theta[i],x0,N)
    #minNLL(P3,x0,NLL,mymin)
    minNLL(PDF,x0,NLL,mymin,args=(time,theta))
       

main()

