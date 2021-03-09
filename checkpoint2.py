'''Checkpoint 2: Part 1: Minimising Chi Squared'''

import numpy as np
import scipy as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Minimise:

    #constructor method
    def __init__(self,filename):
        #convert the text file data into a numpy array
        self.array = np.loadtxt(filename)
        #set x y and error values
        self.x = self.array[:,0]
        self.y = self.array[:,1]
        self.err = self.array[:,2]

    #method to plot data
    def scatter(self):
        plt.scatter(self.x,self.y)
        plt.errorbar(self.x,self.y,yerr = self.err, fmt='o')
        plt.xlabel("x values")
        plt.ylabel("y values")
        plt.show()

    #method to calculate chi squared
    def chi_sq(self,x0):
        m = x0[0]
        c = x0[1]
        yexp = m*self.x + c
        d_sq = ((self.y-yexp)/self.err)**2
        chisq = sum(d_sq)
        return chisq

    #method to minimise chi squared using scipy
    @staticmethod 
    def mymin(fun,x0):
        mx = minimize(fun,x0)
        print('Minimised chi squared is: ',mx["fun"])
        print('Minimised value of m is: ',mx["x"][0])
        print('Minimised value of c is: ',mx["x"][1])
        return mx
'''
    #plot chi squared while varying parameter 'param'
    @staticmethod
    def vary(param,x0):
        xvals = np.arange(param[0],param[1],param[2])
        yvals = np.zeros(len(xvals))  
        for i in range(len(xvals)):
            yvals[i] = data.chi_sq(x0)
        line = np.array([mx["fun"]+1 for i in range(len(m_vals))])
        plt.plot(xvals,yvals,xvals,line,'r--')
        plt.xlabel("parameter")
        plt.ylabel("chi squared")
        plt.title("Chi squared for a range of parameter values")
        plt.show()
'''          
    

def main():
    #define a text file
    filename = 'cp2data.txt'
    #use class to construct data
    data = Minimise(filename)
    #plot data
    data.scatter()
    #set initial values for m and c to 0 and 1 respectively
    #calculate chi squared
    x0 = [0,1]
    data.chi_sq(x0)
    #minimise chi squared
    mx = data.mymin(data.chi_sq,x0)

    #plot chi squared by varying m
    param = [-0.004,-0.0015,0.00001]
    xvals = np.arange(param[0],param[1],param[2])
    yvals = np.zeros(len(xvals))
    #create list and append all values below red line
    error_list = []
    for i in range(len(xvals)):
        x0 = [xvals[i],mx["x"][1]]
        yvals[i] = data.chi_sq(x0)
        if yvals[i] <= mx["fun"]+1:
            error_list.append(xvals[i])
    #plot line of min chi squared +1
    #plot chi squared against m values in a certain range
    line = np.array([mx["fun"]+1 for i in range(len(xvals))])
    plt.plot(xvals,yvals,xvals,line,'r--')
    plt.xlabel("gradient values")
    plt.ylabel("chi squared")
    plt.title("Chi squared for a range of gradient values")
    plt.show()
    
    #plot chi squared by varying c
    param = [0.985,1.005,0.001]
    xvals = np.arange(param[0],param[1],param[2])
    yvals = np.zeros(len(xvals))
    #create list and append all values below red line
    error_list2 = []
    for i in range(len(xvals)):
        x0 = [mx["x"][0],xvals[i]]
        yvals[i] = data.chi_sq(x0)
        if yvals[i] <= mx["fun"]+1:
            error_list2.append(xvals[i])
    #plot line of min chi squared +1
    #plot chi squared against c values in a certain range
    line = np.array([mx["fun"]+1 for i in range(len(xvals))])
    plt.plot(xvals,yvals,xvals,line,'r--')
    plt.xlabel("intercept values")
    plt.ylabel("chi squared")
    plt.title("Chi squared for a range of intercept values")
    plt.show()
    
    '''
    #print intercepts of red line and graphs
    print("The line intercepts the graph at m values: ")
    print(min(error_list), "and" ,max(error_list))
    print("The line intercepts the graph at c values: ")
    print(min(error_list2), "and" ,max(error_list2))
    '''

    #find errors on m and c and print
    #error is the average of two errors
    error_m = 0.5*((mx["x"][0]-min(error_list))+(max(error_list)-mx["x"][0]))
    error_c = 0.5*((mx["x"][1]-min(error_list2))+(max(error_list2)-mx["x"][1]))
    print("The error on m is: ",error_m)
    print("The error on c is: ",error_c)
    
main()
