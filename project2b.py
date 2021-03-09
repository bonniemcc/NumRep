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

def N(tau):
    return 3*np.pi*tau*(1-np.exp(-10/(tau)))

#method uses parameters(F,tau1,tau2) and the time and theta values
#calculates the PDF and returns the NLL as a float
def NLL(x0,time,theta):
    #define parameters
    F,tau1,tau2 = x0[0],x0[1],x0[2]

    PDF = F*(1/N(tau1))*(1+np.cos(theta)*np.cos(theta))*np.exp(-time/tau1) + (1-F)*(1/N(tau2))*3*np.sin(theta)*np.sin(theta)*np.exp(-time/tau2)
    return - np.sum(np.log(PDF))

#when using just the time data, integrate over theta from 0 to 2pi
#this removes theta component and gives a factor of 3pi for each PDF component
def NLL_t(x0,time,theta):
    #define parameters
    F,tau1,tau2 = x0[0],x0[1],x0[2]

    PDF_t = F*(1/N(tau1))*3*m.pi*np.exp(-time/tau1) + (1-F)*(1/N(tau2))*3*m.pi*np.exp(-time/tau2)
    return - np.sum(np.log(PDF_t))

#minNLL minimises a function and parameters x0
#prints minimised float values for each parameter
#returns minimised nll value as a float
def minNLL(fun,x0,time,theta,b):
    nll_min = (minimize(fun,x0,args=(time,theta), bounds=b))["fun"]
    print("Minimised F,tau1,tau2 are: ")
    for i in range(3):
        print((minimize(fun,x0,args=(time,theta), bounds=b))["x"][i])
    return (minimize(fun,x0,args=(time,theta), bounds=b))

#plots each parameter against NLL
def PlotParam(fun,time,theta):
    #define number of points to plot
    num = 1000
    #define list of parameter names
    parameter = ["F","tau1","tau2"]
    F,tau1,tau2
    #define arrays of varied parameters
    F_list = np.linspace(F-0.5,F+0.5,num)
    tau1_list = np.linspace(tau1-0.5,tau1+0.5,num)
    tau2_list = np.linspace(tau2-0.5,tau2+0.5,num)
    x_list = [F_list,tau1_list,tau2_list]
    #create empty array to store
    NLL_list = np.zeros(num)
    #find NLL and plot 
    for j in range(3):
        for i in range(num):
            params = [[F_list[i],1,2],[0.5,tau1_list[i],2],[0.5,1,tau1_list[i]]]
            NLL_list[i] = fun(params[j],time,theta)
        #create empty list to find error in parameter value
        #to store parameter values where the nll is below the horizontal line
        error = []
        for i in range(num):
            if (NLL_list[i]) <= min(NLL_list)+0.5:
                error.append(x_list[j][i])
        err = (max(error)-min(error))/2
        print("The error for ",parameter[j]," is: ",err)
        line = np.array([min(NLL_list)+0.5 for i in range(num)])
        plt.plot(x_list[j],NLL_list,x_list[j],line,'r--')
        plt.xlabel(parameter[j])
        plt.ylabel("NLL")
        plt.show()
    
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

    #minimise NLL by using time data only
    print("Using time data...")
    NLL_t(x0,time,theta)
    minNLL(NLL_t,x0,time,theta,bnds)
    
    #minimise NLL by varying time and theta
    print("Using all data...")
    NLL(x0,time,theta)
    minNLL(NLL,x0,time,theta,bnds)
    
    #finding errors by plotting each parameter against nll
    #vary each parameter while keeping other two parameters fixed
    #use only time data
    PlotParam(NLL_t,time,theta)
    #use time and theta data
    PlotParam(NLL,time,theta)

main()
