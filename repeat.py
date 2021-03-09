'''Class to repeat experiment 500 times and plot distribution'''

import numpy as np
import matplotlib.pyplot as plt
from experiment import Experiment

class Repeat:

    def __init__(self, tau_true):
        self.tau_true = tau_true
        #Create and empty array to store average decay times for each experiment
        self.av_tau = np.empty(500)

    #Function to repeat experiment 500 times
    #Returns a numpy array of average decay times for each experiment
    #Prints the average tau and the error for the first experiment
    def repeat_exp(self,tau_true,num_mu):
        for i in range(500):
            this_exp = Experiment(tau_true,num_mu)
            self.av_tau[i] = this_exp.average()
        print("Average tau for one experiment:",self.av_tau[0],"microseconds")
        print("Error for one experiment:",self.av_tau[0]/(num_mu)**0.5,"microseconds")
        av_av = np.average(self.av_tau)
        stdev = np.std(self.av_tau)
        sterr = stdev /(500)**0.5
        print("Average tau from all experiments:",av_av,"microseconds")
        print("Standard error from all experiments:",sterr,"microseconds")
        return self.av_tau

    #Function to plot distribution of the average muon lifetime from each of the experim$
    def all_plot(self):
        plt.hist(self.av_tau[:],bins=50,density=True)
        plt.title("Distribution of Average Muon Decay Times")
        plt.xlabel("Average decay time (microseconds)")
        plt.ylabel("Normalised Frequency")
        plt.show()
