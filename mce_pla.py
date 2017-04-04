'''

Parallized version to compute evidence from Planck chains
We will analyze all schains in PLA folder


example:

    mpirun -np 6 python mce_pla.py

'''
from __future__ import absolute_import
from __future__ import print_function
import sys
import os
import glob
import numpy as np
import pandas as pd
import astropy as ap
from tabulate import tabulate

from MCEvidence import MCEvidence


if len(sys.argv) > 1:
    verbose=sys.argv[1]
else:
    verbose=0

#-----------------------------
DataSets=['planck_lowl','planck_lowl_lowLike',
           'planck_lowl_lowLike_highL' ,'planck_tauprior',
           'planck_tauprior_highL' ,'WMAP']

ImSamples = ['','post_BAO','post_HST','post_lensing','post_SNLS','post_Union2']


Models={}
Models['model']=['base','base_Alens','base_Alensf','base_alpha1','base_Aphiphi','base_mnu',\
                 'base_mnu_Alens','base_mnu_omegak','base_mnu_w','base_nnu',\
                 'base_nnu_meffsterile','base_nnu_meffsterile_r','base_nnu_mnu',\
                 'base_nnu_r','base_nnu_yhe','base_nrun','base_omegak','base_r','base_w',\
                 'base_w_wa','base_yhe']

Models['npars']=[6,7,7,7,7,7,\
                 7,8,8,7,\
                 8,8,8,\
                 8,8,7,7,7,7,\
                 8,7]

fhandle = open('planck_mce/pla_evidence_table.txt', 'a')

#-----------------------------
from mpi4py import MPI

def mpi_load_balance(MpiSize,nload):
    nmpi_pp=np.zeros(MpiSize,dtype=np.int)
    nmpi_pp[:]=nload/MpiSize
    r=nload%MpiSize
    if r != 0:
        nmpi_pp[1:r-1]=nmpi_pp[1:r-1]+1

    return nmpi_pp

nload=len(DataSets)
mpi_size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
lpp=mpi_load_balance(mpi_size,nload)
        
for ipp in range(lpp[rank]): #data loop
    idd=ipp+lpp[0:rank-1].sum()
    dd=DataSets[idd]
    print('mpi_rank, idd, dd',rank, idd, dd)
    #for dd in DataSets[0:]:
    
    df = pd.DataFrame(index=Models['model'])

    for ss in ImSamples: #importance sample loop
        
        mce=np.zeros(len(Models['model'])) - np.inf
        
        for i,mm in enumerate(Models['model']): #model loop
            
            #print('m,d,m+d+s',mm,dd,ss,type(mm),type(dd),type(ss))
            if ss:
                ff='{}_{}_{}'.format(mm,dd,ss)
            else:
                ff='{}_{}'.format(mm,dd)
                
            method='PLA/{0}/{1}/{2}'.format(mm,dd,ff)
        
            try:
                print('------')
                mce[i] = MCEvidence(method,verbose=verbose).evidence()[0]
                print()
            except:
                print('model+data not available: ',method)
                mce[i]=-np.inf
        
            df[ss]=mce
            
        #get relative evidence
        print('df.data.max=',df[ss].max())
        df[ss]=df[ss]-df[ss].max()
        
    print('--------------- data={}---------'.format(dd))
    print(tabulate(df, headers='keys', tablefmt='psql'))

    #append all tables to file
    fhandle.write('\n')
    fhandle.write('############## data={} ########\n'.format(dd))
    fhandle.write(tabulate(df, headers='keys', tablefmt='psql'))
    fhandle.write('\n')
    #
    df.to_pickle('planck_mce/pla_{}_evidence_df.pkl'.format(dd))
    df.to_latex('planck_mce/pla_{}_evidence_df.tex'.format(dd))


#---------------------------------------------
# Nmodels   = 21
# Ndatasets = 25

# Nparams   = np.zeros(Nmodels)
# Models    = np.empty( (Nmodels), dtype=[('model',object),('npars',int)] )
# DataSets  = np.empty( (Ndatasets), dtype=[('data',object)] )

# DataSets['data'][0]   = 'plikHM_TT_lowTEB'
# DataSets['data'][1]   = 'plikHM_TT_lowTEB_BAO'
# DataSets['data'][2]   = 'plikHM_TT_lowTEB_lensing'
# DataSets['data'][3]   = 'plikHM_TT_tau07'
# DataSets['data'][4]   = 'plikHM_TT_lowl'
# DataSets['data'][5]   = 'plikHM_TT_low1_lensing'
# DataSets['data'][6]   = 'plikHM_TT_lowTEB_reion'
# DataSets['data'][7]   = 'plikHM_TT_lowTEB_lensing_BAO'
# DataSets['data'][8]   = 'plikHM_TTTEEE_lowTEB'
# DataSets['data'][9]   = 'plikHM_TTTEEE_lowl'
# DataSets['data'][10]   = 'plikHM_TTTEEE_lowl_lensing'
# DataSets['data'][11]   = 'plikHM_TTTEEE_lowl_reion'
# DataSets['data'][12]   = 'plikHM_TTTEEE_lowTEB_lensing' 
# DataSets['data'][13]   = 'plikHM_TTTEEE_tau07' 
# DataSets['data'][14]   = 'plikHM_TT_lowTEB_BAO_H070p6JLA'
# DataSets['data'][15]   = 'plikHM_TTTEEE_lowTEB_BAO_H070p6JLA'
# DataSets['data'][16]   = 'WLonlyHeymans'
# DataSets['data'][17]   = 'WLonlyHeymans_BAO'
# DataSets['data'][18]   = 'WLonlyHeymans_BAO_theta'
# DataSets['data'][19]   = 'WLonlyHeymans_BAO_H070p6_theta'
# DataSets['data'][20]   = 'WLonlyHeymans_BAO_H070p6_BAO_theta'
# DataSets['data'][21]   = 'lensonly'
# DataSets['data'][22]   = 'lensonly_BAO'
# DataSets['data'][23]   = 'lensonly_BAO_theta'
# DataSets['data'][24]   = 'lensonly_theta'
