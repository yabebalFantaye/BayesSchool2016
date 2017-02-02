#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
import subprocess
import importlib
import io

import os, sys, math
import numpy as np
import pandas as pd
import sklearn as skl
import statistics
from sklearn.neighbors import NearestNeighbors, DistanceMetric
import matplotlib.pyplot as plt
import scipy.special as sp
from numpy.linalg import inv
from numpy.linalg import det
from functools import reduce

try:
    import getdist
    from getdist import MCSamples, chains, IniFile    
except ImportError:
    print('Consider installing the GetDist module')
    print()
    print('       pip install getdist    ')
    print()

#=======================================

class harry_eg(object):
    def __init__(self,x=None,theta=None,
                 rms=0.2,ptheta=None,verbose=1):
        
        # Generate Data for a Quadratic Function
        if x is None:
            xmin        = 0.0
            xmax        = 4.0
            nDataPoints = 200
            x = np.linspace(xmin, xmax, nDataPoints)
        #data points
        self.x=x
        self.ndata=len(x)
        
        # Data simulation inputs
        if theta is None:
            theta0_true = 1.0
            theta1_true = 4.0
            theta2_true = -1.0
            theta = np.array([theta0_true, theta1_true, theta2_true])
        #parameters
        self.theta=theta
        self.ndim=len(theta)
        
        #flat priors on parameters 
        if ptheta is None:
            ptheta = np.repeat(10.0,self.ndim)

        # Generate quadratic data with noise
        self.y          = self.quadratic(self.theta)
        self.noise_rms = np.ones(self.ndata)*rms
        self.y_sample     = self.y + np.random.normal(0.0, self.noise_rms) 
        
        self.D      = np.zeros(shape = (self.ndata, self.ndim))
        self.D[:,0] = 1.0/self.noise_rms
        self.D[:,1] = self.x/self.noise_rms
        self.D[:,2] = self.x**2/self.noise_rms
        self.b      = self.y_sample/self.noise_rms               
        
        #Initial point to start sampling 
        self.theta_sample=reduce(np.dot, [inv(np.dot(self.D.T, self.D)), self.D.T, self.b])
        
    def quadratic(self,parameters):
        return parameters[0] + parameters[1]*self.x + parameters[2]*self.x**2

    def evidence(self):
        # Calculate the Bayesian Evidence               
        b=self.b
        D=self.D
        #
        num1 = np.log(det(2.0 * np.pi * inv(np.dot(D.T, D))))
        num2 = -0.5 * (np.dot(b.T, b) - reduce(np.dot, [b.T, D, inv(np.dot(D.T, D)), D.T, b]))
        den1 = np.log(self.ptheta.prod()) #prior volume
        #
        log_Evidence = num1 + num2 - den1 #(We have ignored k)
        #
        print('\nThe log-Bayesian Evidence is equal to: {}'.format(log_Evidence))
        
        return log_Evidence
        
    
    def gibbs_dist(self, params, label):
        # The conditional distributions for each parameter
        # This will be used in the Gibbs sampling 
        
        b=self.b
        D=self.D
        sigmaNoise=self.noise_rms
        x=self.x
        ndata=self.ndata
        
        #
        D0 = np.zeros(shape = (ndata, 2)); D0[:,0] = x/sigmaNoise; D0[:,1] = x**2/sigmaNoise 
        D1 = np.zeros(shape = (ndata, 2)); D1[:,0] = 1./sigmaNoise; D1[:,1] = x**2/sigmaNoise 
        D2 = np.zeros(shape = (ndata, 2)); D2[:,0] = 1./sigmaNoise; D2[:,1] = x/sigmaNoise 

        if label == 't0':
            theta_r = np.array([params[1], params[2]])
            v       = 1.0/sigmaNoise
            A       = np.dot(v.T, v)
            B       = -2.0 * (np.dot(b.T, v) - reduce(np.dot, [theta_r.T, D0.T, v]))
            mu      = -B/(2.0 * A)
            sig     = np.sqrt(1.0/A)

        if label == 't1':
            theta_r = np.array([params[0], params[2]])
            v       = x/sigmaNoise
            A       = np.dot(v.T, v)
            B       = -2.0 * (np.dot(b.T, v) - reduce(np.dot, [theta_r.T, D1.T, v]))
            mu      = -B/(2.0 * A)
            sig     = np.sqrt(1.0/A)

        if label == 't2':
            theta_r = np.array([params[0], params[1]])
            v       = x**2/sigmaNoise
            A       = np.dot(v.T, v)
            B       = -2.0 * (np.dot(b.T, v) - reduce(np.dot, [theta_r.T, D2.T, v]))
            mu      = -B/(2.0 * A)
            sig     = np.sqrt(1.0/A)

        return np.random.normal(mu, sig)

    def Sampler(self,nsamples=1000):

        b=self.b
        D=self.D
        
        Niters        = int(nsamples)
        trace         = np.zeros(shape = (Niters, 3))
        logLikelihood = np.zeros(Niters) 

        #previous state
        params=self.theta_sample
        
        for i in range(Niters):
            params[0]  = self.gibbs_dist(params, 't0')
            params[1]  = self.gibbs_dist(params, 't1')
            params[2]  = self.gibbs_dist(params, 't2')
        
            trace[i,:] = params

            logLikelihood[i] = -0.5 * np.dot((b - np.dot(D,trace[i,:])).T, (b - np.dot(D,trace[i,:]))) 

        #save the current state back to theta_sample
        self.theta_sample=params
        
        return trace, logLikelihood 
    
    def info(self):
        return '''Example adabted from Harry's Jupyter notebook. 
        \n{0}-dimensional Polynomial function.'''.format(self.ndim)    

#============================================

class alan_eg(object):
    def __init__(self,ndim=10,ndata=100000,verbose=1):
        #  Generate data

        # Number of dimensions: up to 15 this seems to work OK. 
        self.ndim=ndim

        # Number of data points (not actually very important)
        self.ndata=ndata

        # Some fairly arbitrary mean values for the data.  
        # Standard deviation is unity in all parameter directions.
        std = 1.0
        self.mean  = np.zeros(ndim)
        for i in range(0,ndim):
            self.mean[i]  = np.float(i+1)
              
        # Generate random data all at once:
        self.d2d=np.random.normal(self.mean,std,size=(ndata,ndim))

        # Compute the sample mean and standard deviations, for each dimension
        # The s.d. should be ~1/sqrt(ndata))
        self.mean_sample = np.mean(self.d2d,axis=0)
        self.var_sample  = np.var(self.d2d,axis=0)
        #1sigma error on the mean values estimated from ndata points 
        self.sigma_mean  = np.std(self.d2d,axis=0)/np.sqrt(np.float(ndata))
            
        if verbose>0:
            std_sample  = np.sqrt(self.var_sample)
            print()
            print('mean_sample=',self.mean_sample) 
            print('std_sample=',std_sample)
            print()

    # Compute ln(likelihood)
    def lnprob(self,theta):      
        dM=(theta-self.mean_sample)/self.sigma_mean        
        return (-0.5*np.dot(dM,dM) -
                     self.ndim*0.5*np.log(2.0*math.pi) -
                     np.sum(np.log(self.sigma_mean)))
            
    # Define a routine to generate samples in parameter space:
    def Sampler(self,nsamples=1000):

        # Number of samples:                 nsamples
        # Dimensionality of parameter space: ndim
        # Means:                             mean
        # Standard deviations:               stdev

        
        ndim=self.ndim
        ndata=self.ndata
        mean=self.mean_sample
        sigma=self.sigma_mean
        #
        #Initialize vectors:
        theta = np.zeros((nsamples,ndim))
        f     = np.zeros(nsamples)

        # Generate samples from an ndim-dimension multivariate gaussian:
        theta = np.random.normal(mean,sigma,size=(nsamples,ndim))

        for i in range(nsamples):
            f[i]=self.lnprob(theta[i,:])

        return theta, f   
    def pos(self,n):
        # Generate samples over prior space volume
        return np.random.normal(self.mean_sample,5*self.sigma_mean,size=(n,self.ndim))
    
    def info(self):
        print("Example adabted from Alan's Jupyter notebook") 
        print('{0}-dimensional Multidimensional gaussian.'.format(self.ndim))
        print('ndata=',self.ndata)
        print()
    

    
#===================================

class echain(object):
    # Ntrials:  number of MCMC samples. These are equally-spaced logarithmically between limit given by:
    # powmin, powmax: 10^powmin, 10^powmax are the minimum and maximum numbers of samples.  
    #                 Typical MCMC has ~10^5 samples. powmax=5 is reasonably fast. powmax=6 takes hours.
    # kmax:           The results are plotted using kth-nearest-neighbours, with k between 1 and kmax-1. 
    # method:    a python class with at least __init__(args) and Sampler(nsamples) functions
    # args:      arguments to be passed to method __init__ function
    # verbose:   Chattiness of the run - controls how much information is printed to the screen
    #             0 - very little (only error and warnings)
    #             1 - Essentials for checking sanity of the evidence calculation
    #             2 or more - debugging
    
    def __init__(self,Ntrials = 20,
                      powmin  = 3,
                      powmax  = 4.5,
                      kmax    = 5,        
                      method='alan_eg',
                      args={},
                      verbose=1):
        #
        self.verbose=verbose
        #
        self.powmin=powmin
        self.powmax=powmax
        self.kmax=kmax
        self.Ntrials=Ntrials
        print('my method',method)
        #given a class name, get an instance
        if isinstance(method,str):
            XClass = getattr(sys.modules[__name__], method)
        else:
            XClass=method
            
        if hasattr(XClass, '__class__'):
            print('eknn: method is an instance of a class')
            self.method=XClass
        else:
            print('eknn: method is class variable .. instantiating class')
            self.method=XClass(*args)
           
        #
        self.ndim = self.method.ndim
        
        # The arrays of powers and nchain record the number of samples 
        # that will be analysed at each iteration. 
        #idtrial is just an index
        self.idtrial=np.arange(self.Ntrials)
        self.powers  = np.zeros(self.Ntrials)
        self.nchain  = self.idtrial*0   
        for ipow in self.idtrial:
            dp=(self.powmax-self.powmin)/float(self.Ntrials-1)
            self.powers[ipow] = self.powmin+float(ipow)*dp
            self.nchain[ipow] = 2000#int(pow(10.,self.powers[ipow]))        
            
        try:
            print()
            msg=self.method.info()                        
            print()
        except:
            pass        

    def get_samples(self,nsamples,istart=0):    
        # If we are reading chain, it will be handled here 
        # istart -  will set row index to start getting the samples 
    
        # Generate samples in parameter space by using the passed method        
        samples,fs=self.method.Sampler(nsamples=nsamples)                 
        
        return samples, fs
        
        
    def chains2evidence(self,verbose=None):
        # MLE=maximum likelihood estimate of evidence:
        #
        
        if verbose is None:
            verbose=self.verbose
            
        kmax=self.kmax
        ndim=self.ndim
        
        MLE     = np.zeros((self.Ntrials,kmax+1))

        # Loop over different numbers of MCMC samples (=S):
        itot=0
        for ipow,S in zip(self.idtrial,self.nchain):                
            
            DkNN    = np.zeros((S,kmax+1))
            indices = np.zeros((S,kmax+1))
            volume  = np.zeros((S,kmax+1))
            
            samples,fs=self.get_samples(S,istart=itot)

            # Use sklearn nearest neightbour routine, which chooses the 'best' algorithm.
            # This is where the hard work is done:
            nbrs          = NearestNeighbors(n_neighbors=kmax+1, algorithm='auto').fit(samples)
            DkNN, indices = nbrs.kneighbors(samples)
    
            # Create the posterior for 'a' from the distances (volumes) to nearest neighbour:
            for k in range(1,self.kmax):
                for j in range(0,S):        
                    # Use analytic formula for the volume of ndim-sphere:
                    volume[j,k] = math.pow(math.pi,ndim/2)*math.pow(DkNN[j,k],ndim)/sp.gamma(1+ndim/2)

                # dotp is the summation term in the notes:
                dotp = np.dot(volume[:,k],np.exp(fs))
        
                # The MAP value of 'a' is obtained analytically from the expression for the posterior:
                amax = dotp/(S*k+1.0)
    
                # Maximum likelihood estimator for the evidence (this is normalised to the analytic value):
                MLE[ipow,k] = S*amax
            
            # Output is: for each sample size (S), compute the evidence for kmax-1 different values of k.
            # Final columm gives the evidence in units of the analytic value.
            # The values for different k are clearly not independent. If ndim is large, k=1 does best.
            if verbose>0:
                if ipow==0: 
                    print('(iter,mean,min,max) of LogLikelihood: ',ipow,fs.mean(),fs.min(),fs.max())
                    print('-------------------- useful intermediate parameter values ------- ')
                    print('iter, nsample, dotp, median volume, amax, MLE')
                print(ipow,S,k,dotp,statistics.median(volume[:,k]),amax,MLE[ipow,k])
                
        return MLE


#===============================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        method=sys.argv[1]
    else:
        method=alan_eg
        
    print('Using Class: ',method)
    ealan=echain(method=method,verbose=2)
    ealan.chains2evidence()