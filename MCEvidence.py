#!usr/bin/env python
"""
Version : 0.1.1
Date : 1st March 2017

Authors : Yabebal Fantaye
Email : yabi@aims.ac.za
Affiliation : African Institute for Mathematical Sciences - South Africa
              Stellenbosch University - South Africa

License : MIT

Status : Under Development

Description :
Python2.7 implementation of the evidence estimation from MCMC chains 
as preesented in A. Heavens et. al. 2017
(paper can be found here : https://arxiv.org/abs/ ).
It uses the typical scikit-learn syntax  with a .fit() function for training
and a .predict() function for predictions.

"""

from __future__ import absolute_import
from __future__ import print_function
import subprocess
import importlib
import itertools
from functools import reduce
import io

import os
import sys
import math
import numpy as np
import pandas as pd
import sklearn as skl
import statistics
from sklearn.neighbors import NearestNeighbors, DistanceMetric
import matplotlib.pyplot as plt
import scipy.special as sp
from numpy.linalg import inv
from numpy.linalg import det

from wrap import *

try:
    import getdist
    from getdist import MCSamples, chains, IniFile    
except ImportError:
    print('Consider installing the GetDist module')
    print()
    print('       pip install getdist    ')
    print()

__author__ = "Yabebal Fantaye"
__email__ = "yabi@aims.ac.za"
__license__ = "MIT"
__version__ = "0.1.1"
__status__ = "Development"
    

class echain(object):
    # nbatch:  number of MCMC samples. The sample sizes at each 
    # batch is dertermined from brange: 
    #   if brange= N [constant number] then bscale='constant' 
    #                 and constant sample sizes at each batch
    #   if brange is a list
    #       bscale='lowpower':  equally-spaced logarithmically between limit given by:
    #                           powmin, powmax: 10^powmin, 10^powmax are 
    #                           the minimum and maximum numbers of samples.  
    #       bscale='linear':  equally-spaced linear batch sizes from brange.min() to brange.max()    
    #                 
    # kmax:           The results are plotted using kth-nearest-neighbours, with k between 1 and kmax-1. 
    # method:    a python class with at least __init__(args) and Sampler(nsamples) functions
    # args:      arguments to be passed to method __init__ function
    # verbose:   Chattiness of the run - controls how much information is printed to the screen
    #             0 - very little (only error and warnings)
    #             1 - Essentials for checking sanity of the evidence calculation
    #             2 or more - debugging
    
    def __init__(self,method,nbatch = 1,
                      brange=None,
                      bscale='logpower',
                      kmax    = 5,        
                      args={},                      
                      ischain=True,                      
                      verbose=1,gdkwarg={}):
        #
        self.verbose=verbose
        #
        self.nbatch=nbatch
        self.brange=brange #todo: check for [N] 
        self.bscale=bscale if not isinstance(self.brange,int) else 'constant'
        # The arrays of powers and nchain record the number of samples 
        # that will be analysed at each iteration. 
        #idtrial is just an index
        self.idbatch=np.arange(self.nbatch,dtype=int)
        self.powers  = np.zeros(self.nbatch)
        self.bsize  = np.zeros(self.nbatch,dtype=int)
        self.nchain  = np.zeros(self.nbatch,dtype=int)               
        #
        self.kmax=kmax
        #
        self.ischain=ischain
        #
        self.fname=None
        if ischain:
            if isinstance(method,str):
                print('Using chains: ',method)
                self.fname=method                
                samples = gd.loadMCSamples(method,**gdkwarg)
                npar=6
                self.method={}
                self.method['samples']=samples.samples[:,0:npar]
                self.method['lnprob']=-samples.loglikes
                self.method['weight']=samples.weights                
                #read chain
            else:
                print('dictionary of samples and loglike array passed')
                self.method=method
            #
            self.ndim=self.method['samples'].shape[-1] 
            self.nsample=self.method['samples'].shape[0]
            #print('init minmax logl',method['lnprob'].min(),method['lnprob'].max())            
            print('chain array dimensions: %s x %s ='%(self.nsample,self.ndim))
        else:
            self.nsample=100000
            #given a class name, get an instance
            if isinstance(method,str):
                print('my method',method)
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
         
        #
        self.set_batch()
        #
        if not ischain:    
            try:
                print()
                msg=self.method.info()                        
                print()
            except:
                pass        

    def summary(self):
        print()
        print('ndim={}'.format(self.ndim))
        print('nsample={}'.format(self.nsample))
        print('kmax={}'.format(self.kmax))
        print('brange={}'.format(self.brange))
        print('bsize'.format(self.bsize))
        print('powers={}'.format(self.powers))
        print('nchain={}'.format(self.nchain))
        print()
        
    def get_batch_range(self):
        if self.brange is None:
            powmin,powmax=None,None
        else:
            powmin=np.array(self.brange).min()
            powmax=np.array(self.brange).max()
            if powmin==powmax and self.nbatch>1:
                print('nbatch>1 but batch range is set to zero.')
                raise
        return powmin,powmax
    
    def set_batch(self,bscale=None):
        if bscale is None:
            bscale=self.bscale
        else:
            self.bscale=bscale
            
        #    
        if self.brange is None: 
            self.bsize=self.brange #check
            powmin,powmax=None,None
            self.nchain[0]=self.nsample
            self.powers[0]=np.log10(self.nsample)
        else:
            if bscale=='logpower':
                powmin,powmax=self.get_batch_range()
                self.powers=np.linspace(powmin,powmax,self.nbatch)
                self.bsize = np.array([int(pow(10.0,x)) for x in self.powers])
                self.nchain=self.bsize

            elif bscale=='linear':   
                powmin,powmax=self.get_batch_range()
                self.bsize=np.linspace(powmin,powmax,self.nbatch,dtype=np.int)
                self.powers=np.array([int(log10(x)) for x in self.nchain])
                self.nchain=self.bsize

            else: #constant
                self.bsize=self.brange #check
                self.powers=self.idbatch
                self.nchain=np.array([x for x in self.bsize.cumsum()])
            
    def get_samples(self,nsamples,istart=0,rand=False,thin=True,nthin=None):    
        # If we are reading chain, it will be handled here 
        # istart -  will set row index to start getting the samples 
        
        if self.ischain:
            if rand and not self.brange is None:
                ntot=self.method['samples'].shape[0]
                if nsamples>ntot:
                    print('nsamples=%s, ntotal_chian=%s'%(nsamples,ntot))
                    raise
                idx=np.random.randint(0,high=ntot,size=nsamples)
                samples=self.method['samples'][idx,:]
                fs=self.method['lnprob'][idx]
                self.method['weight'][idx] if 'weight' in self.method else np.ones(nsamples)
            else:
                samples=self.method['samples'][istart:nsamples+istart,:]
                fs=self.method['lnprob'][istart:nsamples+istart]    
                w=self.method['weight'][istart:nsamples+istart] if 'weight' in self.method else np.ones(nsamples)
                #print('get_samples minmax logl',fs.min(),fs.max())
        else:
            # Generate samples in parameter space by using the passed method        
            samples,fs=self.method.Sampler(nsamples=nsamples)                 
            w=np.ones(len(fs))
            
        if thin:
            samples,fs,w=thin_samples(samples,fs,w,nthin=nthin)
                
        return samples, fs,w
        

    def evidence(self,verbose=None,rand=False,profile=False,rprior=1,
                        nproc=-1,prewhiten=True,thin=True,nthin=None):
        # MLE=maximum likelihood estimate of evidence:
        #
        
            
        if verbose is None:
            verbose=self.verbose
            
        kmax=self.kmax
        ndim=self.ndim
        
        MLE     = np.zeros((self.nbatch,kmax))
        
        if profile:
            print('time profiling scikit knn ..')
            profile_data = np.zeros((self.nbatch,2))
        

        # Loop over different numbers of MCMC samples (=S):
        itot=0
        for ipow,nsample in zip(self.idbatch,self.nchain):                
            S=int(nsample)            
            DkNN    = np.zeros((S,kmax+1))
            indices = np.zeros((S,kmax+1))
            volume  = np.zeros((S,kmax+1))
            
            samples_raw,logL,weight=self.get_samples(S,istart=itot,
                                            rand=rand,thin=thin,nthin=nthin)  
            
            # Renormalise loglikelihood (temporarily) to avoid underflows:
            logLmax = np.amax(logL)
            fs    = logL-logLmax
                        
            #print('(mean,min,max) of LogLikelihood: ',fs.mean(),fs.min(),fs.max())
            
            if not unitvar:
                # Covariance matrix of the samples, and eigenvalues (in w) and eigenvectors (in v):
                ChainCov = np.cov(samples_raw.T)
                w,v      = np.linalg.eig(ChainCov)
                Jacobian = math.sqrt(np.linalg.det(ChainCov))

                # Prewhiten:  First diagonalise:
                samples = np.dot(samples_raw,v);

                # And renormalise new parameters to have unit covariance matrix:
                for i in range(ndim):
                    samples[:,i]= samples[:,i]/math.sqrt(w[i])
            else:
                #no diagonalisation
                Jacobian=1
                samples=samples_raw

            # Use sklearn nearest neightbour routine, which chooses the 'best' algorithm.
            # This is where the hard work is done:
            if profile:
                with Timer() as t:
                    nbrs          = NearestNeighbors(n_neighbors=kmax+1, 
                                                     algorithm='auto',n_jobs=nproc).fit(samples)
                    DkNN, indices = nbrs.kneighbors(samples)
                                    
                profile_data[ipow,0]=S
                profile_data[ipow,1]=t.secs                    
            else:
                nbrs          = NearestNeighbors(n_neighbors=kmax+1, 
                                                 algorithm='auto',n_jobs=nproc).fit(samples)
                DkNN, indices = nbrs.kneighbors(samples)                
    
            # Create the posterior for 'a' from the distances (volumes) to nearest neighbour:
            for k in range(1,self.kmax):
                for j in range(0,S):        
                    # Use analytic formula for the volume of ndim-sphere:
                    volume[j,k] = math.pow(math.pi,ndim/2)*math.pow(DkNN[j,k],ndim)/sp.gamma(1+ndim/2)
                
                
                #print('volume minmax: ',volume[:,k].min(),volume[:,k].max())
                #print('weight minmax: ',weight.min(),weight.max())
                
                # dotp is the summation term in the notes:
                dotp = np.dot(volume[:,k]/weight[:],np.exp(fs))
        
                # The MAP value of 'a' is obtained analytically from the expression for the posterior:
                amax = dotp/(S*k+1.0)
    
                # Maximum likelihood estimator for the evidence (this is normalised to the analytic value):
                SumW     = np.sum(weight)
                #print('SumW*S*amax*Jacobian',SumW,S,amax,Jacobian)
                MLE[ipow,k] = math.log(SumW*S*amax*Jacobian) + logLmax
            
                # Output is: for each sample size (S), compute the evidence for kmax-1 different values of k.
                # Final columm gives the evidence in units of the analytic value.
                # The values for different k are clearly not independent. If ndim is large, k=1 does best.
                if self.brange is None:
                    #print('(mean,min,max) of LogLikelihood: ',fs.mean(),fs.min(),fs.max())
                    print('k={},nsample={}, dotp={}, median_volume={}, a_max={}, MLE={}'.format( 
                        k,S,dotp,statistics.median(volume[:,k]),amax,MLE[ipow,k]))
                
                else:
                    if verbose>0:
                        if ipow==0: 
                            print('(iter,mean,min,max) of LogLikelihood: ',ipow,fs.mean(),fs.min(),fs.max())
                            print('-------------------- useful intermediate parameter values ------- ')
                            print('nsample, dotp, median volume, amax, MLE')                
                        print(S,k,dotp,statistics.median(volume[:,k]),amax,MLE[ipow,k])
         
        if self.brange is None:
            MLE=MLE[0,1:]
         
        print()
        print('MLE[k=(1,2,3,4)] = ',MLE)
        print()
        
        if profile:
            return (MLE, profile_data)
        else:  
            return MLE
    
    def vis_mle(self,MLE,figsize=(15,15),**kwargs):
        #visualise 
        kmax=self.kmax
        ndim=self.ndim
        nchain=self.nchain #.cumsum()
        powers=self.powers

        fig,ax=plt.subplots(figsize=figsize)
        plt.subplot(2,2,1)
        plt.plot(powers,np.log10(MLE[:,1:kmax]))
        plt.xlabel('log N')
        plt.ylabel('$log(\hat E / E_{true})$')
        plt.title('Evidence ratio')
        plt.legend(['k=%s'%k for k in range(1,kmax+1)])

        plt.subplot(2,2,2)
        plt.plot(nchain,MLE[:,1:kmax])
        plt.xlabel('N')
        plt.ylabel('$\hat E / E_{true}$')
        plt.title('Evidence ratio')

        plt.subplot(2,2,3)
        plt.plot(1.0/nchain,MLE[:,1:kmax])
        plt.xlabel('1/N')
        plt.ylabel('$\hat E / E_{true}$')
        plt.title('Evidence ratio')

        plt.subplot(2,2,4)
        plt.plot(pow(nchain,-1.0/ndim),MLE[:,1:kmax])
        plt.xlabel('$1/N^{1/d}$')
        plt.ylabel('$\hat E / E_{true}$')
        plt.title('Evidence ratio')

        plt.tight_layout()        

           

#===============================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        method=sys.argv[1]
    else:
        method=alan_eg
        
    print('Using Class: ',method)
    ealan=echain(method=method,verbose=2)
    ealan.chains2evidence()
