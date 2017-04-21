from __future__ import absolute_import
from __future__ import print_function
import sys
import os
import glob
import numpy as np
import pandas as pd
from tabulate import tabulate
import pickle
import fileinput
from mpi4py import MPI
from argparse import ArgumentParser
import logging

#---------------------------------------------------
#------- Path and sub-directory folders ------------
#---------------------------------------------------
rootdir='COM_CosmoParams_fullGrid_R2.00'

#list of cosmology parameters
cosmo_params=['omegabh2','omegach2','theta','tau','omegak','mnu','meffsterile','w','wa',
              'nnu','yhe','alpha1','deltazrei','Alens','Alensf','fdm','logA','ns','nrun',
              'nrunrun','r','nt','ntrun','Aphiphi']
    
# Types of model to consider. Below a more
# comprehensive list is defined using wildcards.
# The function avail_data_list() extracts all data names
# available in the planck fullgrid directory.
DataSets=['plikHM_TT_lowTEB','plikHM_TT_lowTEB_post_BAO','plikHM_TT_lowTEB_post_lensing','plikHM_TT_lowTEB_post_H070p6','plikHM_TT_lowTEB_post_JLA','plikHM_TT_lowTEB_post_zre6p5','plikHM_TT_lowTEB_post_BAO_H070p6_JLA','plikHM_TT_lowTEB_post_lensing_BAO_H070p6_JLA','plikHM_TT_lowTEB_BAO','plikHM_TT_lowTEB_BAO_post_lensing','plikHM_TT_lowTEB_BAO_post_H070p6','plikHM_TT_lowTEB_BAO_post_H070p6_JLA','plikHM_TT_lowTEB_lensing','plikHM_TT_lowTEB_lensing_post_BAO','plikHM_TT_lowTEB_lensing_post_zre6p5','plikHM_TT_lowTEB_lensing_post_BAO_H070p6_JLA','plikHM_TT_tau07plikHM_TT_lowTEB_lensing_BAO','plikHM_TT_lowTEB_lensing_BAO_post_H070p6','plikHM_TT_lowTEB_lensing_BAO_post_H070p6_JLA','plikHM_TTTEEE_lowTEB','plikHM_TTTEEE_lowTEB_post_BAO','plikHM_TTTEEE_lowTEB_post_lensing','plikHM_TTTEEE_lowTEB_post_H070p6','plikHM_TTTEEE_lowTEB_post_JLA','plikHM_TTTEEE_lowTEB_post_zre6p5','plikHM_TTTEEE_lowTEB_post_BAO_H070p6_JLA','plikHM_TTTEEE_lowTEB_post_lensing_BAO_H070p6_JLA','plikHM_TTTEEE_lowl_lensing','plikHM_TTTEEE_lowl_lensing_post_BAO','plikHM_TTTEEE_lowl_lensing_post_BAO_H070p6_JLA','plikHM_TTTEEE_lowTEB_lensing']


# Types of model to consider. Below a more
# comprehensive list is defined using wildcards.
# The function avail_model_list() extracts all data names
# available in the planck fullgrid directory.
Models={}
Models['model']=['base','base_omegak','base_Alens','base_Alensf','base_nnu','base_mnu',\
                 'base_nrun','base_r','base_w','base_alpha1','base_Aphiphi','base_yhe',\
                 'base_mnu_Alens','base_mnu_omegak','base_mnu_w','base_nnu_mnu',\
                 'base_nnu_r','base_nrun_r','base_nnu_yhe','base_w_wa',\
                 'base_nnu_meffsterile','base_nnu_meffsterile_r']

#---------------------------------------
#-------- define some useful functions -
#---------------------------------------
def avail_data_list(mm):
    '''
    Given model name, extract all available data names
    '''    
    l=glob.glob( '{0}/{1}/*/*_1.txt'.format(rootdir,mm) )
    l1=[x.split('_1')[0] for x in l]
    l2=[x.split('base_')[1] for x in l1]
    return l1,l2

def avail_model_list(dd,nmax=0,sorter=Models['model']):
    '''
    Given data name, extract all available models
    If sorter is not None, sorting will be based
    according to the order of sorter
    '''
    df=pd.DataFrame()
    l=glob.glob( '{0}/*/*/*_{1}_1.txt'.format(rootdir,dd) )
    df['l1']=[x.split('_1')[0] for x in l]    
    df['l2']=df['l1'].apply(lambda x:x.split('/')[1])
    
    #sort df based on sorter order
    if sorter:
        df['l2'] = df['l2'].astype("category")
        df['l2'].cat.set_categories(sorter, inplace=True)    
    df=df.sort_values('l2')

    if nmax>0:
        df=df.iloc[0:nmax]
    return df['l1'].values,df['l2'].values

def iscosmo_param(p,l=cosmo_params):
    '''
    check if parameter 'p' is cosmological or nuisance
    '''
    return p in l

def params_info(fname):
    '''
    Extract parameter names, ranges, and prior space volume
    from CosmoMC *.ranges file
    '''
    par=np.genfromtxt(fname+'.ranges',dtype=None,names=('name','min','max'))#,unpack=True)
    parName=par['name']
    parMin=par['min']
    parMax=par['max']
    
    parMC={'name':[],'min':[],'max':[],'range':[]}
    for p,pmin,pmax in zip(parName, parMin,parMax):
        if not np.isclose(pmax,pmin) and iscosmo_param(p):
            parMC['name'].append(p)
            parMC['min'].append(pmin)
            parMC['max'].append(pmax)
            parMC['range'].append(np.abs(pmax-pmin))
    #
    parMC['str']=','.join(parMC['name'])
    parMC['ndim']=len(parMC['name'])
    parMC['volume']=np.array(parMC['range']).prod()
    
    return parMC

#----------------------------------------------------------
#------- define which model, data etc. list to be used ----
#----------------------------------------------------------
dd='plikHM_TTTEEE_lowTEB'
path_list, name=avail_model_list(dd,nmax=1)
