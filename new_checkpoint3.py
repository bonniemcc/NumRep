'''Checkpoint 3: Solving First Order ODEs for the PN Junction
    using the Euler and 4th Order Runge-Kutta (RK4) methods''' 

import math as m
import numpy as np
import matplotlib.pyplot as plt
#Use Class: ChargeDistribution.py
from ChargeDistribution import ChargeDistribution
    #Has two useful methods: q=evaluate(x) and plot show()

#Define useful methods

#Method for finding charge values for set x values
#Using class method evaluate(x)
def Charge(x_vals,data):
    rho_vals = np.zeros(len(x_vals))
    for i in range(len(x_vals)):
        rho_vals[i] = data.evaluate(x_vals[i])
    return rho_vals

#Euler method function
def Euler(x_vals,delta,f,arg):
    y_vals = np.zeros(len(x_vals))
    for i in range(len(x_vals)-1):
        y_vals[i+1] = y_vals[i] + delta*f(arg[i])
    return y_vals
    
#RK4 method function
def RK4(x_vals,delta,k):
    z_vals = np.zeros(len(x_vals))
    for i in range(len(x_vals)-1):
        z_vals[i+1] = z_vals[i] + delta*(k[0]/6+k[1]/3+k[2]/3+k[3]/6)
    return z_vals

#Plot function for Euler and RK4 Method
def plot(x_vals,y_vals,z_vals,title):
    plt.plot(x_vals,y_vals, x_vals, z_vals, 'r')
    plt.xlabel("x")
    plt.ylabel(str(title))
    plt.title(str(title)+ " Distribution")
    plt.show()

def function(argument):
    func = argument
    return func

def main():
    #Initialise class data
    data = ChargeDistribution()
    
    #Choose delta and set x values between -2 and 2
    delta = float( input( "Enter step size (delta) "))
    x_vals = np.arange(-2,2+delta,delta)

    #Part 0: Calculate charge values and plot charge distribution
    Q = Charge(x_vals,data)
    data.show('Charge Distribution')
    
    #Part 1: Determine E(x) in the range [-2,2]
    
    #Implement Euler method
    y1 = Euler(x_vals,delta,data.evaluate,x_vals)
    #Implement RK4 method
    k1 = np.zeros(len(x_vals)-1)
    for i in range(len(x_vals)-1):
        k1[0] = data.evaluate(x_vals[i])
        k1[1] = data.evaluate(x_vals[i]+delta/2)
        k1[2] = k1[1]
        k1[3] = data.evaluate(x_vals[i]+delta)
    z1 = RK4(x_vals,delta,k1)
    #Plot E(x) for both methods
    plot(x_vals,y1,z1,"Electric Field")

    #Part 2: Determine Voltage in the range [-2,2]
    
    #Need to calculate new E values for twice as many x values
    new_x = np.arange(-2,2+delta,delta/2)
    #new_q = Charge(new_x,data)
    #Find new E values using Euler method
    #Step size is half the size of delta
    new_y = Euler(new_x,delta/2,data.evaluate,new_x)
    
    #Implement Euler method
    y2 = Euler(x_vals,delta,y1,function)
    #Implement RK4 method
    k2 = np.zeros(len(x_vals)-1)
    for i in range(len(x_vals)-1):
        k2[0] = new_y[2*i]
        k2[1] = new_y[2*i-1]
        k2[2] = k2[1]
        k2[3] = new_y[2*(i+1)]
    z2 = RK42(x_vals,delta,k2)
    #Plot V(x) for both methods
    plot(x_vals,-y2,-z2,"Voltage")
    
main()
