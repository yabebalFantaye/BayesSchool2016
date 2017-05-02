# coding: utf-8

import os, sys, math, csv

# These are not necessary if your libraries 
#  are set up in the right places: (getdist needed in Yabe's code)
#sys.path.append('/usr/local/bin')
#sys.path.append('/Users/Dropbox/Evidence')


# In[2]:

import numpy as np
#import pandas as pd
import sklearn as skl
import statistics
from sklearn.neighbors import NearestNeighbors, DistanceMetric
#from sklearn.neighbors import KDTree
#import matplotlib.pyplot as plt
import scipy.special as sp
from scipy import random, linalg
#from scipy.stats import multivariate_normal as multivariate_normal



# In[3]:

# Basic simulation parameters:

# kth NN:  Max k+1:
kmax    = 2

# Number of repeated chains for each of the Planck models:
nchains = 4

# Thinning level:
thinningFraction    = 0.1
thinning            = False

DoAll = False


# In[4]:

def Evidence(samples,logL,LogLmax,S,weight,TotalWeight,priorvol,Npars,kmax=2):

# MLE=maximum likelihood estimate of evidence: MLEPW: with pre-whitening
# powers and nchain record the number of samples (just for plotting).

    MLE     = np.zeros(kmax+1)
    MLEPW   = np.zeros(kmax+1)

    DkNN      = np.zeros((S,kmax+1))
    indices   = np.zeros((S,kmax+1))
    volume    = np.zeros((S,kmax+1))
    
    DkNNPW    = np.zeros((S,kmax+1))
    indicesPW = np.zeros((S,kmax+1))
    volumePW  = np.zeros((S,kmax+1))
    
# Use sklearn nearest neighbour routine, which chooses the 'best' algorithm.
# This is where the hard work is done:

    nbrs          = NearestNeighbors(n_neighbors=kmax+1, algorithm='auto').fit(samples)
    DkNN, indices = nbrs.kneighbors(samples)
    
# For prewhitening, we need the covariance matrix of the samples, 
# and eigenvalues (in w) and eigenvectors (in v):

    ChainCov = np.cov(samples.T)
    w,v      = np.linalg.eig(ChainCov)
    Jacobian = math.sqrt(np.linalg.det(ChainCov))
    
# Prewhiten:  First diagonalise:

    samplesPW = np.dot(samples,v);
    
# And renormalise new parameters to have unit covariance matrix:

    for i in range(Npars):
        samplesPW[:,i]= samplesPW[:,i]/math.sqrt(w[i])

# Covariance matrix of samplesPW should be the identity:

#    print(np.cov(samplesPW.T))    

# Redo evidence calculation with prewhitened parameters to compare:

    nbrsPW            = NearestNeighbors(n_neighbors=kmax+1, algorithm='auto').fit(samplesPW)
    DkNNPW, indicesPW = nbrsPW.kneighbors(samplesPW)

# Create the posterior for 'a' from the distances (volumes) to nearest neighbour:

    for k in range(1,kmax):
        for j in range(0,S):
        
# Use analytic formula for the volume of Npars-sphere:
            volume[j,k]   = math.pow(math.pi,Npars/2)*math.pow(DkNN[j,k],  Npars)/sp.gamma(1+Npars/2)
            volumePW[j,k] = math.pow(math.pi,Npars/2)*math.pow(DkNNPW[j,k],Npars)/sp.gamma(1+Npars/2)

# dotp is the summation term in the notes:
        dotp   = np.dot(volume[:,k]/weight[:],  np.exp(logL))
        dotpPW = np.dot(volumePW[:,k]/weight[:],np.exp(logL))
        
# The MAP value of 'a' is obtained analytically from the expression for the posterior:
        amax   = dotp/(S*k+1.0)
        amaxPW = dotpPW/(S*k+1.0)
    
# Maximum likelihood estimator for the evidence:

# CHANGED FOR H0 PRIOR:

#        SumW     = np.sum(weight)
        SumW     = np.sum(TotalWeight)
    
        MLE[k]   = SumW*amax
        MLEPW[k] = SumW*amaxPW*Jacobian 

        print(type(MLEPW[k]),MLEPW[k])
# Output is: for each sample size (S), compute the evidence for kmax-1 different values of k.
# Final columm gives the evidence in units of the analytic value.
# The values for different k are clearly not independent. If Npars is large, k=1 does best.

# The (extra) priorvolume is also passed, so the prior (assumed uniform) provides a weight:

        logpriorvol   = math.log(priorvol)
        print('** k,SumW,S, amax,Jacobian, logpriorvol,LogLmax,MLE',k,SumW,S,amaxPW,Jacobian,logpriorvol,LogLmax,math.log(MLEPW[k])   +LogLmax - logpriorvol)
        print('----')
# Reinstate the maximum amplitude of the likelihood:

        
#    print(MLE[1:kmax])
#    print(MLEPW[1:kmax])
    
    if(np.amin(MLE[1:kmax])>0.0):
        logEvidence   = math.log(MLE[1:kmax])   +LogLmax - logpriorvol
    else:
        logEvidence   = -200000.0 * np.ones(kmax-1)
        print(MLE[1:kmax])
        
    if(np.amin(MLEPW[1:kmax])>0.0):      
        logEvidencePW = math.log(MLEPW[1:kmax]) +LogLmax - logpriorvol
    else:
        logEvidencePW = -200000.0 * np.ones(kmax-1)
        print(MLEPW[1:kmax])
        
    return logEvidence,logEvidencePW

def H0prior(h):
    
# From Riess et al 2016:
    sigma_h = 1.74
    hbar    = 73.24
    h0prior = math.exp(-0.5*math.pow((h-hbar)/sigma_h,2))/(math.sqrt(2.*math.pi)*sigma_h)
    return h0prior


# In[5]:

MLEvidence   = np.zeros((nchains+1,kmax))
MLEvidencePW = np.zeros((nchains+1,kmax))

# Read Planck chains and extract the relevant pieces:

# Open file

Directory ='/Users/alanheavens/Documents/Research/Planck Evidence/'
Directory ='./'
ModelDir  = 'COM_CosmoParams_fullGrid_R2.00/'

Nmodels      = 21
Ndatasets    = 38
Npost        = 11
Nalldatasets = 1000
h0prior      = True

# Post is the list of postprocessing datasets used for some Planck chains.

# Declare some arrays and dictinaries, although the latter is probably overkill:

Nparams     = np.zeros(Nmodels)
Models      = np.empty( (Nmodels), dtype=[('model',object),('npars',int),('priorvolume',float)] )
DataSets    = np.empty( (Ndatasets), dtype=[('data',object),('nmodels',int),('H0',int)] )
Post        = np.empty( (Npost), dtype=[('post',object)])
AllDataSets = np.empty( (Nalldatasets), dtype=[('data',object),('nmodels',int)] )
LogEvidence = np.zeros((Nmodels,Nalldatasets))
Error       = np.zeros((Nmodels,Nalldatasets))

LogEvidence[:,:] = -100000.0

# Datasets and models:
# Not sure of the most elegant way to do these... 

DataSets['data'][0]   = 'plikHM_TT_lowTEB'
DataSets['data'][1]   = 'plikHM_TT_lowTEB_BAO'
DataSets['data'][2]   = 'plikHM_TT_lowTEB_lensing'
DataSets['data'][3]   = 'plikHM_TT_tau07'
DataSets['data'][4]   = 'plikHM_TT_lowl'
DataSets['data'][5]   = 'plikHM_TT_low1_lensing'
DataSets['data'][6]   = 'plikHM_TT_lowTEB_reion'
DataSets['data'][7]   = 'plikHM_TT_lowTEB_lensing_BAO'
DataSets['data'][8]   = 'plikHM_TT_lowEB'
DataSets['data'][9]   = 'plikHM_TTTEEE_lowTEB'
DataSets['data'][10]   = 'plikHM_TTTEEE_lowl'
DataSets['data'][11]   = 'plikHM_TTTEEE_lowl_lensing'
DataSets['data'][12]   = 'plikHM_TTTEEE_lowl_reion'
DataSets['data'][13]   = 'plikHM_TTTEEE_lowTEB_lensing' 
DataSets['data'][14]   = 'plikHM_TTTEEE_tau07' 
DataSets['data'][15]   = 'plikHM_TT_lowTEB_BAO_H070p6_JLA'
DataSets['data'][16]   = 'plikHM_TTTEEE_lowTEB_BAO_H070p6_JLA'
DataSets['data'][17]   = 'plikHM_TTTEEE_lowTEB_nnup39_BAO'
DataSets['data'][18]   = 'plikHM_TTTEEE_lowTEB_nnup57_BAO'
DataSets['data'][19]   = 'WLonlyHeymans'
DataSets['data'][20]   = 'WLonlyHeymans_BAO'
DataSets['data'][21]   = 'WLonlyHeymans_BAO_theta'
DataSets['data'][22]   = 'WLonlyHeymans_BAO_H070p6_theta'
DataSets['data'][23]   = 'WLonlyHeymans_BAO_H070p6_BAO_theta'
DataSets['data'][24]   = 'lensonly'
DataSets['data'][25]   = 'lensonly_BAO'
DataSets['data'][26]   = 'lensonly_BAO_theta'
DataSets['data'][27]   = 'lensonly_theta'
DataSets['data'][28]   = 'plikHM_EE'
DataSets['data'][29]   = 'plikHM_lensing'
DataSets['data'][30]   = 'plikHM_lowEB'
DataSets['data'][31]   = 'plikHM_lowTEB'
DataSets['data'][32]   = 'plikHM_TE'
DataSets['data'][33]   = 'plikHM_TE_lensing'
DataSets['data'][34]   = 'plikHM_TE_lowEB'
DataSets['data'][35]   = 'plikHM_TE_lowTEB'
DataSets['data'][36]   = 'plikHM_TT_WMAPTEB'
DataSets['data'][37]   = 'WMAP'

DataSets['H0']=[22,22,22,22,22,22,22,22,22,34,34,34,34,34,34,22,34,34,34,6,6,5,5,5,6,6,5,5,               14,14,14,14,15,15,15,15,22,8]

# m labels the models:

Models['model']=['base','base_omegak','base_Alens','base_Alensf','base_nnu','base_mnu',                 'base_nrun','base_r','base_w','base_alpha1','base_Aphiphi','base_yhe',                 'base_mnu_Alens','base_mnu_omegak','base_mnu_w','base_nnu_mnu',                 'base_nnu_r','base_nnu_yhe','base_w_wa',                 'base_nnu_meffsterile','base_nnu_meffsterile_r']

Models['npars']=[6,7,7,7,7,7,                 7,7,7,7,7,7,                 8,8,8,8,                 8,8,8,                 8,9]

# Extra prior volume for the new parameters (from Planck 2013 Cosmo. Pars paper XVI):

Models['priorvolume'] = [1, 0.6, 10., 10., 9.95, 5.,                        2., 3., 4., 2., 10., 0.4,                        50., 3., 20., 49.75,                        29.85, 3.98, 20.0,                        29.85, 89.55]

Post['post']=['','_post_BAO','_post_lensing','_post_H070p6','_post_JLA','_post_zre6p5',              '_post_BAO_H070p6','_post_H070p6_JLA','_post_BAO_H070p6_JLA',              '_post_lensing_H070p6_JLA','_post_lensing_BAO_H070p6_JLA']

DataSets['nmodels'][:]=0

# Create a log file to check things such as the right number of parameters have been selected:
logfile = open("temp/PlanckEvidenceLogfile.txt","w")

# Append resultS to file:
fileout = open("temp/PlanckEvidenceResults.txt", "a")
    
# Don't need to do all of the models.  Select here:
# Note that some of the datasets (e.g. BAO) are *sometimes* bolted on to existing chains
# so the chains are importance-sampled.  This comes in the weight assigned.

if(DoAll):
    datasetmin = 0
    datasetmax = Ndatasets
    modelmin   = 0
    modelmax   = Nmodels
    postmax    = Npost
else:
    datasetmin = 0
    datasetmax = 1 #Ndatasets
    modelmin   = 0
    modelmax   = 10
    postmax    = 1 #Npost

model_list=range(modelmin,modelmax)
model_list=[0,1]
for m in model_list:
    print('\nModel',m,':', Models['model'][m])
    print('========================================')
    
    Model    = Models['model'][m]
    Npars    = Models['npars'][m]
    priorvol = Models['priorvolume'][m]
    
    countdataset = 0

    for d in [9]: #range(datasetmin,datasetmax):
        print('\nDataset',d,':', DataSets['data'][d])
        print('----------------------------------------------')
    
        DataDir = DataSets['data'][d]
        
# WLonlyHeymans has tau and theta fixed, so reduce the number of parameters by 2:

        if(DataSets['data'][d]=='WLonlyHeymans_BAO_theta'):
            Npars    = Models['npars'][m]-2
            print('Number of parameters for WL_BAO_theta =',Npars)
        elif(DataSets['data'][d]=='WLonlyHeymans_BAO'):
            Npars    = Models['npars'][m]-1
            print('Number of parameters for WL_BAO =',Npars)
        elif(DataSets['data'][d]=='WLonlyHeymans'):
            Npars    = Models['npars'][m]-1
            print('Number of parameters for WL =',Npars)
        elif(DataSets['data'][d]=='lensonly'):
            Npars    = Models['npars'][m]-1
            print('Number of parameters for lensonly =',Npars)
        elif(DataSets['data'][d]=='lensonly_BAO'):
            Npars    = Models['npars'][m]-1
            print('Number of parameters for lensonly =',Npars)
        elif(DataSets['data'][d]=='lensonly_BAO_theta'):
            Npars    = Models['npars'][m]-2
            print('Number of parameters for lensonly_BAO_theta =',Npars)
        elif(DataSets['data'][d]==DataSets['data'][d]=='lensonly_theta'):
            Npars    = Models['npars'][m]-2
            print('Number of parameters for lensonly_theta =',Npars)                    
        elif(DataSets['data'][d].startswith('plikHM_TTTEEE_lowTEB_nnup39_BAO')):
            Npars    = Models['npars'][m]-1
            print('Number of parameters for plikHM_TTTEEE_lowTEB_nnup39_BAO =',Npars)
            
# Add in postprocessing options:
        print('---------------------------------------')
        print('model, data: ',DataDir, Model)
        print('***** Npars, priorvol',Npars, priorvol)
        print('---------------------------------------')
        
        for pp in range(postmax):
            PostExt = Post['post'][pp]

            for chain in range(1,nchains+1):
                MLEvidence[chain,1:]   = 0
                MLEvidencePW[chain,1:] = 0
                meanMLEPW              = 0
                stdMLEPW               = 0

# Create filename from model and dataset:

                File          = Model+'_'+DataDir+PostExt+'_'+str(chain)+'.txt'
                PlanckFile    = Directory+ModelDir+Model+'/'+DataDir+'/'+File

# Create filename containing input parameters:

                parametersfile = Model+'_'+DataDir+'.paramnames'                
                ParamsFile     = Directory+ModelDir+Model+'/'+DataDir+'/'+parametersfile
                
                ShortFileName = File

# Calculate Evidence if the file exists:

                if(os.path.isfile(PlanckFile)):
            
                    FileExists = True               
                    RawDataTable = np.loadtxt(PlanckFile)                    
# Thin the chain:

                    RawS         = len(RawDataTable)
# Thin the chain at random.  Since some points are weighted, we draw from a 
# Poisson distribution with mean thinningFraction * weight, and make this the new weight.
# Some points may end up with weight>1 after thinning, but if the thinningFraction is << 1,
# they will be rare.  Note that some chains have weights in the 20s and maybe higher.

# Raw weights are in RawDataTable[:,0]
# Do this in a pedestrian way, for safety.

                    newWeight = np.zeros(RawS)

                    if(thinning):
#                        thinningRandoms = np.random.rand(RawS)
                        for j in range(RawS):
                            oldWeight = RawDataTable[j,0]
                            newWeight[j] = np.random.poisson(oldWeight*thinningFraction)
                            RawDataTable[j,0] = float(newWeight[j])
#                            print(int(oldWeight),newWeight)
#                        DataTable       = RawDataTable[np.where(thinningRandoms<thinningFraction)]
                        DataTable   = RawDataTable[np.where(newWeight>0)]
                    else:
                        DataTable  = RawDataTable
                        
                    S       = len(DataTable)
                    
                    if(S==0 or (thinning and pp>0)):
                        print('Thinned chain is empty, or a postprocessed chain')
                        continue
                    else:
                        samples = np.zeros((S,Npars))
                        logL    = np.zeros(S)
                        weight  = np.zeros(S)
                        TotalWeight = np.ones(S)

                        weight[:]          =  DataTable[:,0]
                        logL[:]            = -DataTable[:,1]
                        samples[:,0:Npars] =  DataTable[:,2:2+Npars]
                        
# Now find the H0 line:                        
                        if(chain==1 and os.path.isfile(ParamsFile)):
                            params = open(ParamsFile,"r")

                            linenumber = 0
                            for line in params:
                                linenumber += 1
                                if line.startswith('H0'):
                                    h0col = linenumber+1
                                    print('H0 is line ',h0col)
                            params.close()
                    
                        if(h0prior):
#                            h0col = DataSets['H0'][d]
                            for i in range(S):
                                TotalWeight[i] = weight[i]*H0prior(DataTable[i,h0col])
                            print('H0 limits:',np.amin(DataTable[:,h0col]),np.amax(DataTable[:,h0col]))    
#================================================================================
# Back to my code, using the Evidence method:

# Renormalise likelihood (temporarily) to avoid underflows:

                        logLmax = np.amax(logL)
                        logL    = logL-logLmax
                        MLEvidence[chain,1:],MLEvidencePW[chain,1:]=Evidence(samples,logL,logLmax,S,weight,TotalWeight,priorvol,Npars,kmax)
                        print(len(DataTable),MLEvidence[chain,1:],MLEvidencePW[chain,1:])
                
# Save some of the parameters to check:

                        if(chain==1 and os.path.isfile(ParamsFile)):
                            params = open(ParamsFile,"r")
                            nlines = 0
                            logfile.write(ShortFileName+'\n')
                            for line in params:
                                if(nlines>Npars-2):
                                    logfile.write(line)
                                if(nlines==Npars):
                                    break
                                nlines += 1
                            params.close()                                    
                else:
                    FileExists = False
                    if(chain==1):
                        print(ShortFileName,' does not exist')
                            
            if(FileExists and ((not thinning) or (pp==0))):
                meanMLEPW = np.mean(MLEvidencePW[1:nchains+1,1])
                stdMLEPW  = np.std(MLEvidencePW[1:nchains+1,1])/math.sqrt(nchains)
        
                LogEvidence[m,countdataset] = meanMLEPW
                Error[m,countdataset]       = stdMLEPW
                
                AllDataSets['nmodels'][countdataset] += 1
                    
                print(ShortFileName,': ln(E) =',meanMLEPW,' +/-',stdMLEPW)

# Count the data set even if a chain is not present for this model:

            AllDataSets['data'][countdataset]=DataDir+PostExt                            
            countdataset +=1
            
            fileout.write(Model+' '+DataDir+PostExt+': ln(E) ='+str(meanMLEPW)+' +/-'+str(stdMLEPW)+'\n')
               
fileout.close()

# We'll store the evidence values only for datasets that have at least two models evaluated:

FinalDataSets    = np.where(AllDataSets['nmodels']>1)

# Write output in csv file (for exploring in Excel!):

if(thinning):
    datafile = open('temp/PlanckEvidenceThinnedRiessH0prior.csv', 'w')
else:
    datafile = open('temp/PlanckEvidenceNotThinnedRiessH0prior.csv', 'w')   

writer = csv.writer(datafile)
writer.writerow(AllDataSets['data'][np.where(AllDataSets['nmodels']>1)])

FinalLogEvidence = LogEvidence[0:modelmax,np.where(AllDataSets['nmodels']>1)]
    
# MLE values:

for m in range(modelmax):
    writer.writerow(np.ndarray.flatten(FinalLogEvidence[m,:]))
    
# Write the model names in a column so they can be put in the spreadsheet in the right place:

for m in range(modelmax):
    writer.writerow([Models['model'][m]])

FinalError = Error[0:modelmax,np.where(AllDataSets['nmodels']>1)]

# Errors:    

writer.writerow(AllDataSets['data'][np.where(AllDataSets['nmodels']>1)])

for m in range(modelmax):
    writer.writerow(np.ndarray.flatten(FinalError[m,:]))

# Compute Bayes factor w.r.t. STANDARD or BEST model:

BayesFactor = np.zeros((Nmodels,(np.ndarray.flatten(FinalLogEvidence[0,:])).size))

writer.writerow(AllDataSets['data'][np.where(AllDataSets['nmodels']>1)])
for m in range(modelmax):
    BayesFactor[m,:] = FinalLogEvidence[m,:] - np.amax(FinalLogEvidence,axis=0)
#    BayesFactor[m,:] = FinalLogEvidence[m,:] - FinalLogEvidence[0,:]
    
# Models that are not computed are given a large negative evidence.  This cludge
# sets them to -100000:

for m in range(modelmax):
    BayesFactor[np.where(BayesFactor<-90000.)] = -100000.
    writer.writerow(BayesFactor[m,:])
                    
datafile.close()  
logfile.close()

print('Data written to file',datafile)


