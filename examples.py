
# coding: utf-8

# In[1]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

#%load_ext memory_profiler
#%load_ext line_profiler


# In[26]:

#use case example
#%lprun -f range range(100000)
#%mprun >>

# with Timer() as t:
#     range(1000)
# print("elasped lpush: %s s" % t.secs)


# In[3]:

from __future__ import print_function
import IPython
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

get_ipython().magic(u'matplotlib inline')


# ## PyStan example

# In[4]:

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
    if alan_fit in locals():
        #faster as we don't need C++ compiling
        alan_fit = pystan.stan(fit=alan_fit, data=mvgauss_dat, 
                               iter=100000, chains=4)     
    else:
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


# In[34]:

#
gdstans=samples2gdist(alan_stan_chain['samples'],alan_stan_chain['lnprob'],
                     trueval=dmv.mean,px='m')
gdstans.corner()


# In[ ]:

# Here given pystan samples and log probability, we compute evidence ratio 
ealan=eknn.echain(method=alan_stan_chain,verbose=2,ischain=True,brange=[3,5])
MLE,ptime=ealan.chains2evidence(rand=True,profile=True) 
ealan.vis_mle(MLE)


# In[6]:

# plot KNN timing profile
fig,ax=plt.subplots(figsize=(15,6))
plt.plot(np.log10(ptime[:,0]),ptime[:,1])
plt.xlabel('log N')
plt.ylabel('Time in Seconds')
plt.title('Scikit KNN time profile')
#plt.legend(['k=%s'%k for k in range(1,kmax+1)])


# ### Pystan harry example 

# In[21]:

import bayesglm as sglm
import pystan

harry_stanmodel='''
 data {
         int<lower=1> K;
         int<lower=0> N;
         real y[N];
         matrix[N,K] x;
 }
 parameters {
         vector[K] beta;
         real sigma;
 }
 model {         
         real mu[N];
         vector[N] eta   ;
         eta <- x*beta;
         for (i in 1:N) {
            mu[i] <- (eta[i]);
         };
         increment_log_prob(normal_log(y,mu,sigma));

 }
 '''   
harry=eknn.harry_eg()
df=pd.DataFrame()
df['x1']=harry.x
df['x2']=harry.x**2
df['y']=harry.y_sample

harry_data={'N':harry.ndata,
           'K':harry.ndim,
            'x':df[['x1','x2']],
           'y':harry.y_sample}
# Intialize pystan -- this will convert our pystan code into C++
# and run MCMC
#harry_fit = pystan.stan(model_code=harry_stanmodel, data=harry_data,
#                  iter=1000, chains=4)


# In[23]:

iterations=10000
class jeffry_prior():
    def __init__(self, sigma):
        self.sigma = sigma

    def __repr__(self):
        return self.to_string()

    def to_string(self):
        return "normal(0,{0})".format(self.sigma)
class normal_prior():
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return self.to_string()

    def to_string(self):
        return "normal({0},{1})".format(self.mu, self.sigma)


#priors=(((i+1,), jeffry_prior(np.sqrt(harry.ndata))) for i in range(harry.ndim-1) )

priors={"x%s"%(i+1) : normal_prior(harry.theta[i+1],0.2) for i in range(harry.ndim-1) }

cache_fn='chains/harry_pystan_chain.pkl'
#read chain from cache if possible 
try:
    raise
    print('reading chain from: '+cache_fn)
    harry_stan_chain = pickle.load(open(cache_fn, 'rb'))
except:
    harry_fit = sglm.stan_glm("y ~ x1 + x2", df, 
                              family=sglm.family.gaussian(), 
                              iterations=iterations) #,priors=priors

    # Extract PyStan chain for Harry's GLM example
    harry_stan_chain=harry_fit.extract(permuted=True)   
    print('writing chain in: '+cache_fn)
    with open(cache_fn, 'wb') as f:
            pickle.dump(harry_stan_chain, f)

    #print stan model
    harry_model=harry_fit.stanmodel.model_code.expandtabs() #.rsplit('\n') 
    with open('harry.stan', 'w') as f:   
        f.write(harry_model[:])

theta_means = harry_stan_chain['beta'].mean(axis=0)
print('estimated: ',theta_means)
print('input: ',harry.theta)

#np.testing.assert_allclose(theta_means, harry.theta, atol=.01)


# In[27]:

plt.plot(harry_stan_chain['lnprob'])
harry_stan_chain['lnprob']=harry_stan_chain['lnprob']+2*np.log(0.1/np.sqrt(1.0*harry.ndata))
plt.plot(harry_stan_chain['lnprob'])


# In[24]:

# Check input parameter recovery and estimate evidence
if 'beta' in harry_stan_chain.keys(): harry_stan_chain['samples']=harry_stan_chain.pop('beta')
if 'lp__' in harry_stan_chain.keys(): harry_stan_chain['lnprob']=harry_stan_chain.pop('lp__')
print(harry_stan_chain['samples'].shape)

#
gdstans=samples2gdist(harry_stan_chain['samples'],harry_stan_chain['lnprob'],
                     trueval=harry.theta,px='\\theta')
gdstans.corner(figsize=(10,10))
#gdstans.labels


# In[28]:

# Here given pystan samples and log probability, we compute evidence ratio 
eharry=eknn.echain(method=harry_stan_chain,verbose=2,ischain=True)
MLE=eharry.chains2evidence() 
eharry.vis_mle(MLE)


# In[ ]:




# ## Emcee example

# In[164]:

##learn about emcee sampler using help
#help(mec2d.sampler)


# ## emcee sampling using N-dimensional Gaussian likelihood

# In[210]:

#
#gd_mc.samples.getName()


# In[21]:

#Evidence calculation based on emcee sampling
mNd=eknn.alan_eg()
mecNd=make_emcee_chain(mNd,nwalkers=300)
samples,lnp=mecNd.mcmc(nmcmc=50000,thin=50)


# In[26]:

#corner plot can be done also using getdist wrapper
#getdist wrapper has a lot more functionality than just plotting
gd_mc=samples2gdist(samples,lnp,trueval=mNd.mean,px='m')
print('correlation length:',gd_mc.samples.getCorrelationLength(3))
gd_mc.samples.thin(20)
##gd_mc.corner()
#mecNd.emcee_sampler.get_autocorr_time(fast=True)


# In[28]:

thin_samples=gd_mc.samples.samples
thin_lnp=gd_mc.samples.loglikes

print(len(thin_lnp),thin_samples.shape)

#estimate evidence
ealan=eknn.echain(method={'samples':thin_samples,'lnprob':thin_lnp},
                  verbose=2,ischain=True,brange=[3,4.2])
MLE=ealan.chains2evidence(rand=True) 


# In[29]:

ealan.vis_mle(MLE)


# ## Emcee 2D example

# In[ ]:

#test model class .. visualise uniform sampling
m2d=eknn.model_2d()

#test emcee wrapper 
mec2d=make_emcee_chain(m2d,nwalkers=200)
chain2d,fs=mec2d.mcmc(nmcmc=500)


#let's trangle plot chain samples 
fig = corner.corner(chain2d, labels=["$m$", "$b$"], extents=[[-1.1, -0.8], [3.5, 5.]],
                      truths=m2d.p, quantiles=[0.16, 0.5, 0.84], 
                    show_titles=True, labels_args={"fontsize": 40})
fig.set_size_inches(10,10)

# Plot back the results in the space of data
#fig = plt.figure()
#xl = np.array([0, 10])
#for m, b in chain2d[np.random.randint(len(chain2d), size=100)]:
#    if m<0:
#        plt.plot(xl, m*xl+b, color="k", alpha=0.1)
    
#plt.plot(xl, m2d.p[0]*xl+m2d.p[1], color="r", lw=2, alpha=0.8)
#plt.errorbar(m2d.x, m2d.y, yerr=m2d.yerr, fmt=".k")
#plt.title('Input Data vs Samples (grey)')
#fig.set_size_inches(12, 8)


# In[ ]:




# In[ ]:


