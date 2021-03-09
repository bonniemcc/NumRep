'''Project Parts 2&3: 
Import time and theta values from datafile.txt.
Perform a 2-dimensional maximum likelihood fit to determine the
best fit values of the parameters (F, tau1, tau2) and their errors.
Perform the fit using the time data only and then using the time
and theta data
Project Part 4:
Calculate the proper errors by varying parameter about best fit value.
Re-perform minimisation of all other parameters at each point.
'''

import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#define method to find normalisation function
def N(tau):
    return 3*np.pi*tau*(1-np.exp(-10/(tau)))

#method uses parameters(F,tau1,tau2) and the time and theta values
#calculates the PDF and returns the NLL as a float
def NLL(x0,time,theta):
    #define parameters
    F,tau1,tau2 = x0

    PDF = F*(1/N(tau1))*(1+np.cos(theta)*np.cos(theta))*np.exp(-time/tau1) 
                + (1-F)*(1/N(tau2))*3*np.sin(theta)*np.sin(theta)*np.exp(-time/tau2)
    return - np.sum(np.log(PDF))

#when using just the time data, integrate over theta from 0 to 2pi
#this removes theta component and gives a factor of 3pi for each PDF component
def NLL_t(x0,time,theta):
    #define parameters
    F,tau1,tau2 = x0

    PDF_t = F*(1/N(tau1))*3*m.pi*np.exp(-time/tau1) 
                + (1-F)*(1/N(tau2))*3*m.pi*np.exp(-time/tau2)
    return - np.sum(np.log(PDF_t))

#minNLL minimises a function and parameters x0
#prints minimised float values for each parameter
#returns minimised nll value as a float
def minNLL(fun,x0,time,theta,b):
    nll_min = (minimize(fun,x0,args=(time,theta), bounds=b))["fun"]
    return (minimize(fun,x0,args=(time,theta), bounds=b))["x"]

def ParamPlot(NLLfunc,time,theta,x0):
    #define number of points to plot
    num = 1000
    #define list of parameter names
    parameter = ["F","tau1","tau2"]
    #define parameters
    F,tau1,tau2 = x0
    #define parameter lists
    F_list = np.linspace(F-0.1,F+0.1,num)
    tau1_list = np.linspace(tau1-0.1,tau1+0.1,num)
    tau2_list = np.linspace(tau2-0.1,tau2+0.1,num)
    x_list = [F_list,tau1_list,tau2_list]
    #create empty array to store y-values
    NLL_list = np.zeros(num)
    #loop over 3 parameters
    for j in range(3):
        for i in range(num):
            params = [[F_list[i],tau1,tau2],[F,tau1_list[i],tau2],[F,tau1,tau2_list[i]]]
            NLL_list[i] = NLLfunc(params[j],time,theta)
        #create empty list to find error in parameter value
        #to store parameter values where the nll is below the horizontal line
        error = []
        for i in range(num):
            if (NLL_list[i]) <= min(NLL_list)+0.5:
                error.append(x_list[j][i])
        err = (max(error)-min(error))/2
        print("   The error for ",parameter[j]," is: ",err)
        line = np.array([min(NLL_list)+0.5 for i in range(num)])
        plt.plot(x_list[j],NLL_list,x_list[j],line,'r--')
        plt.xlabel(parameter[j])
        plt.ylabel("NLL")
        plt.show()

def ParamProper(NLLfunc,time,theta,x0,minNLL,b):
    #define number times to vary each parameter
    num = 100
    #define list of parameter names
    parameter = ["F","tau1","tau2"]
    #define parameters
    F,tau1,tau2 = x0
    #define parameter lists
    F_list = np.linspace(F-0.1,F+0.1,num)
    tau1_list = np.linspace(tau1-0.1,tau1+0.1,num)
    tau2_list = np.linspace(tau2-0.1,tau2+0.1,num)
    x_list = [F_list,tau1_list,tau2_list]
    #create empty array to store NLL values
    NLL_list = np.zeros(num)
    for j in range(3):
        for i in range(num):
            params = [[F_list[i],tau1,tau2],[F,tau1_list[i],tau2],[F,tau1,tau2_list[i]]]
            NLL_list[i] = NLLfunc(params[j],time,theta)
            #re-minimize at each point
            params[j] = minNLL(NLLfunc,params[j],time,theta,b)
        #create empty list to find error in parameter value
        #to store parameter values where the nll is below the horizontal line
        error = []
        for i in range(num):
            if (NLL_list[i]) <= min(NLL_list)+0.5:
                error.append(x_list[j][i])
        err = (max(error)-min(error))/2
        print("   The error for ",parameter[j]," is: ",err)


def main():
    
    #get data by reading text file
    data = np.loadtxt(open('datafile-Xdecay.txt', 'r'))
    #define n, the number of data points
    n = len(data)
    #set time and theta values as two columns of text file
    time,theta = data[:,0],data[:,1]
    #set parameter values (F,tau1,tau2)
    x0 = [1,1,1]
    #set bounds of parameters
    bnds = ((0,1.0),(0,np.inf),(0,np.inf))

    #first use only the time data
    
    print("Using time data...")
    #minimise NLL
    NLL_t(x0,time,theta)
    #set new parameters 
    x1 = minNLL(NLL_t,x0,time,theta,bnds)
    print("Minimised F,tau1,tau2 are: ",x1)
    #find errors by plotting each parameter against nll
    #vary each parameter while keeping other two parameters fixed
    ParamPlot(NLL_t,time,theta,x1)
    print("Part 4...")
    #find proper errors by re-minimising nll at each point
    #set the new parameters each time
    ParamProper(NLL_t,time,theta,x1,minNLL,bnds)
    
    #now use the full time and theta data
    
    #minimise NLL
    print("Using all data...")
    NLL(x0,time,theta)
    #set new parameters 
    x2 = minNLL(NLL,x0,time,theta,bnds)
    print("Minimised F,tau1,tau2 are: ",x2)
    #find errors by plotting each parameter against nll
    #vary each parameter while keeping other two parameters fixed
    ParamPlot(NLL,time,theta,x2)
    print("Part 4...")
    #find proper errors by re-minimising nll at each point
    #set the new parameters each time
    ParamProper(NLL,time,theta,x2,minNLL,bnds)
    

main()