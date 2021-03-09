'''Class to run experiment which finds the decay time of 1000 muons'''

import numpy as np
import matplotlib.pyplot as plt

class Experiment:

    #Initialise muons and generate 1000 random values drawn from an exponential distribution
    def __init__(self,tau_true,num_mu):
        self.tau_true = tau_true
        self.random = np.random.exponential(tau_true,num_mu)

    #Function which returns the average lifetime as a float
    def average(self):
        #Estimate the lifetime of the muon by taking the mean average
        return np.average(self.random)

    #Function which calculates the difference between the average lifetime and the true value
    def difference(self, tau_true):
        #Compare estimate to the true lifetime
        self.dif = self.av - tau_true

    #Function which plots the histogram of decay times of 1000 muons
    def exp_plot(self):
        plt.hist(self.random[:],bins=50,density=True)
        plt.title("Decay Times of 1000 muons")
        plt.xlabel("Decay Time (microseconds)")
        plt.ylabel("Normalised Frequency")
        plt.show()
