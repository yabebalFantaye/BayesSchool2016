import pickle
#
import os, sys, math
import pandas as pd
import time
import numpy as np
#import pandas as pd
import sklearn as skl
import statistics
from sklearn.neighbors import NearestNeighbors, DistanceMetric
#from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import scipy.special as sp
#
import eknn
from wrap import *

#pretty plots if seaborn is installed
try: 
    import seaborn as sns
    sns.set(style='ticks', palette='Set2',font_scale=1.5)
    #sns.set() 
except:
    pass

import pystan

# First we will write our Baysian model using
# Stan code. 
# We will then define our data and call pystan 
# to do MCMC sampling using NUT sampler
# 
mvgauss_code="""
data {
  int<lower=1> N;
  vector[N] v;
  vector[N] y;
}
parameters {
  vector[N] mu; 
}
transformed parameters {  
  cov_matrix[N] Sigma;
  for (i in 1:N) 
    for (j in 1:N)
      Sigma[i,j] <- 0 + if_else(i==j, v[i], 0.0);
}
model {
  increment_log_prob(multi_normal_log(y,mu,Sigma));
  // y ~ multi_normal(mu,Sigma); //drops constants
}
"""

# Define data model: we will use 10-dimensional Gaussian
# as given in the alan_eg
dmv=eknn.alan_eg()
mvgauss_dat={'y':dmv.mean_sample,
             'v':dmv.sigma_mean**2,
             'N':dmv.ndim}

#file name where to save/read chain
cache_falan='chains/alan_pystan_chain.pkl'

#read chain from cache if possible 
try:
    #raise
    print('reading chain from: '+cache_falan)
    alan_stan_chain = pickle.load(open(cache_falan, 'rb'))
except:
    # Get pystan chain-- this will convert our pystan code into C++
    # and run MCMC
    try:
        #faster as we don't need C++ compiling
        alan_fit = pystan.stan(fit=alan_fit, data=mvgauss_dat, 
                               iter=100000, chains=4)     
    except:
        alan_fit = pystan.stan(model_code=mvgauss_code, data=mvgauss_dat,
                      iter=100000, chains=4)    
    

    # Extract PyStan chain for Harry's GLM example
    alan_stan_chain=alan_fit.extract(permuted=True)   
    print('writing chain in: '+cache_falan)
    with open(cache_falan, 'wb') as f:
            pickle.dump(alan_stan_chain, f)
            
if 'mu' in alan_stan_chain.keys(): alan_stan_chain['samples']=alan_stan_chain.pop('mu')
if 'lp__' in alan_stan_chain.keys(): alan_stan_chain['lnprob']=alan_stan_chain.pop('lp__')
    
print('chain shape: ',alan_stan_chain['samples'].shape)

#convert pystan chains to Getdist object 
gdstans=samples2gdist(alan_stan_chain['samples'],alan_stan_chain['lnprob'],
                     trueval=dmv.mean,px='m')

#------------------------------------------------------
#  Evidence ratio estimate and time profile
#------------------------------------------------------

if not os.path.exists('output'):
    os.makedirs('output')
if not os.path.exists('figures'):
    os.makedirs('figures')

try:
    MLE=np.loadtxt('output/alan_mle.txt')
    ptime=np.loadtxt('output/alan_t_vs_logn_profile.txt')
except:
    # Here given pystan samples and log probability, we compute evidence ratio 
    ealan=eknn.echain(method=alan_stan_chain,verbose=2,
                          ischain=True,brange=[3,3.5])
    MLE,ptime=ealan.chains2evidence(rand=True,profile=True,nproc=-1) #use all procs 
    ealan.vis_mle(MLE)

    #save MLE
    np.savetxt('output/alan_mle.txt',MLE)
    np.savetxt('output/alan_t_vs_logn_profile.txt',ptime)

# plot KNN timing profile
print('plotting Scikit kNN time profile')
fig,ax=plt.subplots(figsize=(15,6))
plt.plot(np.log10(ptime[:,0]),ptime[:,1],
             color='k',
             marker='o',
             markersize=20,
             markerfacecolor='r',
             fillstyle='full')
plt.xlabel('log N')
plt.ylabel('Time in Seconds')
plt.title('Scikit kNN time profile')
plt.savefig('figures/alan_t_vs_logN.pdf')
#plt.legend(['k=%s'%k for k in range(1,kmax+1)])

#------------------------------------
#  test as a function of #processors
#------------------------------------
#
from sklearn.neighbors import NearestNeighbors, DistanceMetric

print('analysing sckit knn #CPU scaling')

kmax=5
nproc_list=[1,5,10,15,20,25,30,35,40,45]
#nproc_list=[1,2]

try:
    profile_nproc=np.loadtxt('output/alan_t_vs_nproc_profile.txt')
except:
    profile_nproc = np.zeros((len(nproc_list),2))
    for ipow,nproc in enumerate(nproc_list):
        print('using nproc=%s'%nproc)
        with Timer() as t:
            nbrs = NearestNeighbors(n_neighbors=kmax+1,
                                    algorithm='auto',
                                    n_jobs=nproc).fit(alan_stan_chain['samples'][::5,:])
        
            DkNN, indices = nbrs.kneighbors(alan_stan_chain['samples'][::5,:])

        print('elapsed time: %f ms' % t.secs)
        
    profile_nproc[ipow,0]=nproc
    profile_nproc[ipow,1]=t.secs
    np.loadtxt('output/alan_t_vs_nproc_profile.txt',profile_nproc)
    
# plot KNN timing profile
print('plotting sckit knn #CPU scaling')
fig,ax=plt.subplots(figsize=(15,6))
plt.plot(np.log10(profile_nproc[:,0]),profile_nproc[:,1],
             color='k',
             marker='o',
             markersize=20,
             markerfacecolor='r',
             fillstyle='full')
plt.xlabel('Number of CPUs')
plt.ylabel('Time in Seconds')
plt.title('Scikit kNN #CPUs scaling')
plt.savefig('figures/alan_t_vs_nproc_scaling.pdf')
    
