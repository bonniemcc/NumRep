'''Project:
Part 1: Generate 10,000 random events and plot time and theta distributions.
Part 2: Perform maximum likelihood fit to determine optimum parameter values
        and their errors using the time data only.
Part 3: Perform maximum likelihood fit to determine optimum parameter values
        and their errors using the time and theta data.
Part 4: Calculate the proper errors by varying each parameter about the best
        fit value and re-performing the minimisation of all other parameters
        at each point.'''

import math as m
import numpy as np
import matplotlib.pyplot as plt
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize

def N(tau):
    return 3*np.pi*tau*(1-np.exp(-10/(tau)))

#Define method to find PDF
#The PDF of the decay is given by a fraction F of P1 and the remainder of P2
#The first PDF is when F=1 and the second is when F=0
def PDF(time,theta,x0):
    F,tau1,tau2 = x0
    PDF1 = F*(1/N(tau1))*(1+np.cos(theta)**2)*np.exp(-time/tau1)
    PDF2 = (1-F)*(1/N(tau2))*3*(np.sin(theta)**2)*np.exp(-time/tau2)
    return (PDF1 + PDF2)

#Define method to find PDF when using time data only
#To get PDF in terms of time only, integrate over theta between 0 and 2pi
def PDF_t(time,theta,x0):
    F,tau1,tau2 = x0
    return F*(1/N(tau1))*3*m.pi*np.exp(-time/tau1) + (1-F)*(1/N(tau2))*3*m.pi*np.exp(-time/tau2)

def random_gen(PDF,x0):
    #Maximum value for the normalised PDF is y2
    #Loop through random values and append if function > y2
    #Set initial values for y and y2
    y = 0.
    y2 = 1.
    while (y2 > y) :
        rand_t = np.random.uniform(0,10)
        rand_theta = np.random.uniform(0,2*m.pi)
        y = PDF(rand_t,rand_theta,x0)
        y2 = np.random.uniform()

    return rand_t, rand_theta

#Define function to plot each PDF with two variables
def plot2D(time,theta,x0,function,equation):
    x,y = meshgrid(time,theta)
    Z = function(x,y,x0)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, Z, rstride=1, cstride=1, 
                      cmap=cm.RdBu,linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    title(equation)
    plt.show()

#Function to plot PDFs
def plotPDF(variable,xlabel):
    plt.hist(variable,bins=100,density=True)
    plt.title("Normalised histogram")
    plt.xlabel(xlabel)
    plt.ylabel("Normalised Frequency")
    plt.show()

#method uses parameters(F,tau1,tau2) and the time and theta values
#calculates the PDF and returns the NLL as a float
def NLL(x0,time,theta):
    #define parameters
    F,tau1,tau2 = x0[0],x0[1],x0[2]
    PDF1 = F*(1/N(tau1))*(1+np.cos(theta)**2)*np.exp(-time/tau1)
    PDF2 = (1-F)*(1/N(tau2))*3*(np.sin(theta)**2)*np.exp(-time/tau2)
    return - np.sum(np.log(PDF1+PDF2))

#when using just the time data, integrate over theta from 0 to 2pi
#this removes theta component and gives a factor of 3pi for each PDF component
def NLL_t(x0,time,theta):
    #define parameters
    #F,tau1,tau2 = x0[0],x0[1],x0[2]

    #PDF_t = F*(1/N(tau1))*3*m.pi*np.exp(-time/tau1) + (1-F)*(1/N(tau2))*3*m.pi*np.exp(-time/tau2)
    return - np.sum(np.log(PDF_t(time,theta,x0)))

#minNLL minimises a function and parameters x0
#prints minimised float values for each parameter
#returns minimised nll value as a float
def minNLL(func,x0,time,theta,b):
    nll_min = (minimize(func,x0,args=(time,theta), bounds=b))["fun"]
    print("Minimised F,tau1,tau2 are: ")
    for i in range(3):
        print((minimize(func,x0,args=(time,theta), bounds=b))["x"][i])
    return (minimize(func,x0,args=(time,theta), bounds=b))
    
#plots each parameter against NLL and finds errors
def PlotParam(fun,time,theta,func, x0):
    #num is the number of points to plot
    num = 1000
    #define list of parameter names
    parameter = ["F","tau1","tau2"]

    F, tau1, tau2 = x0
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

#plots each parameter against NLL and finds errors
def PlotParamProper(fun,time,theta,func,minfun,b):
    #num is the number of points to plot
    num = 1000
    #define list of parameter names
    parameter = ["F","tau1","tau2"]
    #define arrays of varied parameters
    F_list = np.arange(0.3,0.6,0.3/(num))
    tau1_list = np.arange(1.4,1.9,0.5/(num))
    tau2_list = np.arange(1.8,2.4,0.6/(num))
    x_list = [F_list,tau1_list,tau2_list]
    #create empty array to store
    NLL_list = np.zeros(num)
    #loop over each parameter
    for j in range(len(x_list)):
        for i in range(num):
            x0 = [[F_list[i],1,2],[0.5,tau1_list[i],2],[0.5,1,tau1_list[i]]]
            #calculate the NLL
            NLL_list[i] = fun(params[j],time,theta,N,func)
            #re-minimise and set new parameters
            x0[j][j+1] = minfun(fun,x0[j],time,theta,func,b)["x"][j+1]
            x0[j][j+2] = minfun(fun,x0[j],time,theta,func,b)["x"][j+2]
            x0[j][j-1] = minfun(fun,x0[j],time,theta,func,b)["x"][j+1]
            x0[j][j-2] = minfun(fun,x0[j],time,theta,func,b)["x"][j+2]    
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
     
    #Set parameters, x0 = [F,tau1,tau2]
    n = 10000
    x0 = [0.5,1.0,2.0]

    '''
    #Create arrays of two observed quantities
    #Use 100 equally spaced values 
    time = np.arange(0,10,0.1)
    theta = np.arange(0,2*m.pi,2*m.pi/100)

    #Plot both normalised 2D PDFs (F=1,F=0) and total PDF (F=0.5)
    plot2D(time,theta,[1,1,2],N,PDF,
            '$P_1=(1/N_1)(1+cos^2 theta) exp[-t/ tau_1]$')
    plot2D(time,theta,[0,1,2],N,PDF,
            '$P_2=(1/N_2)(3sin^2 theta) exp[-t/ tau_2]$')
    plot2D(time,theta,[0.5,1,2],N,PDF,
            '$P_T=F.P_1 + (F-1).P_2$')

    #Draw time and theta values from PDFs
    time = np.zeros(n)
    theta = np.zeros(n)
    for i in range(n):
        time[i],theta[i] = random_gen(PDF,x0,N)

    #Plot distribution of time and then theta
    plotPDF(time,"time")
    plotPDF(theta,"theta")
    '''
    #get data by reading text file
    data = np.loadtxt(open('datafile-Xdecay.txt', 'r'))
    #define n, the number of data points
    n = len(data)
    #set time and theta values as two columns of text file
    time,theta = data[:,0],data[:,1]
    #set parameter values (F,tau1,tau2)
    x0 = [1,1,1]
    print("x0 is ... " ,x0)
    #set bounds of parameters
    bnds = ((0,1.0),(0,10),(0,10))
    
    #minimise NLL by using time data only
    print("Using time data...")
    #HELP!!!!!!!
    
    NLL_t(x0,time,theta)
    optresults = minNLL(NLL_t,x0,time,theta,bnds)
    print(optresults)
    
    #minimise NLL by varying time and theta
    print("Using all data...")
    NLL(x0,time,theta)
    optresults = minNLL(NLL,x0,time,theta,bnds)
    print(optresults)

    x0 = [optresults["x"][0], optresults["x"][1], optresults["x"][2]]
    
    #finding errors by plotting each parameter against nll
    #vary each parameter while keeping other two parameters fixed
    #use only time data
    PlotParam(NLL_t,time,theta,PDF_t, x0)
    #use time and theta data
    PlotParam(NLL,time,theta,PDF, x0)

    #find proper errors by reminimising parameters at each point
    #use only time data
    PlotParamProper(NLL_t,time,theta,PDF_t,minNLL,bnds)
    #use time and theta data
    PlotParamProper(NLL,time,theta,PDF,minNLL,bnds)


main()
