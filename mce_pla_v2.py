'''
Planck MCMC chains evidence analysis. The data is available from [1].

Parameters
---------

Parallized version to compute evidence from Planck chains
We will analyze all schains in PLA folder

Returns
---------

The code writes results to terminal as well as a file. The default path
to the output files is

.. path:: planck_mce_fullGrid_R2/

Notes
---------

The full analysis using a single MPI process takes about ~30mins.


Examples
---------

To run the evidence estimation using 6 MPI processes

.. shell:: mpirun -np 6 python mce_pla.py

References
-----------

.. [1] Fullgrid Planck MCMC chains:
http://irsa.ipac.caltech.edu/data/Planck/release_2/ancillary-data/cosmoparams/COM_CosmoParams_fullGrid_R2.00.tar.gz


'''
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
from MCEvidence import MCEvidence

#-------------- IO ---

if len(sys.argv) > 1:
    kmax=int(sys.argv[1])
else:
    kmax=2

assert isinstance(kmax,int),'kmax must be int'
assert kmax >= 2,'kmax must be >=2'

if len(sys.argv) > 2:
    verbose=sys.argv[2]
else:
    verbose=0

#
#-----------------------------
from mpi4py import MPI

def mpi_load_balance(MpiSize,nload):
    nmpi_pp=np.zeros(MpiSize,dtype=np.int)
    nmpi_pp[:]=nload/MpiSize
    r=nload%MpiSize
    if r != 0:
        nmpi_pp[1:r-1]=nmpi_pp[1:r-1]+1

    return nmpi_pp
#
mpi_size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
comm = MPI.COMM_WORLD

#amode=MPI.MODE_WRONLY
#fhandle = MPI.File.Open(comm, fout, amode)

#----------------------------------
#------------- IO dirs ------------
#
rootdir='COM_CosmoParams_fullGrid_R2.00'
#list of cosmology parameters
cosmo_params=['omegabh2','omegach2','theta','tau','omegak','mnu','meffsterile','w','wa',
              'nnu','yhe','alpha1','deltazrei','Alens','Alensf','fdm','logA','ns','nrun',
              'nrunrun','r','nt','ntrun','Aphiphi']
#output idr        
outdir='planck_mce_fullGrid_R2'
fout_df='{}/mce_{}.txt'
if not os.path.exists(outdir):
    os.makedirs(outdir)

#interesting subfolders
DataSets=['plikHM_TT_lowTEB','plikHM_TT_lowTEB_post_BAO','plikHM_TT_lowTEB_post_lensing','plikHM_TT_lowTEB_post_H070p6','plikHM_TT_lowTEB_post_JLA','plikHM_TT_lowTEB_post_zre6p5','plikHM_TT_lowTEB_post_BAO_H070p6_JLA','plikHM_TT_lowTEB_post_lensing_BAO_H070p6_JLA','plikHM_TT_lowTEB_BAO','plikHM_TT_lowTEB_BAO_post_lensing','plikHM_TT_lowTEB_BAO_post_H070p6','plikHM_TT_lowTEB_BAO_post_H070p6_JLA','plikHM_TT_lowTEB_lensing','plikHM_TT_lowTEB_lensing_post_BAO','plikHM_TT_lowTEB_lensing_post_zre6p5','plikHM_TT_lowTEB_lensing_post_BAO_H070p6_JLA','plikHM_TT_tau07plikHM_TT_lowTEB_lensing_BAO','plikHM_TT_lowTEB_lensing_BAO_post_H070p6','plikHM_TT_lowTEB_lensing_BAO_post_H070p6_JLA','plikHM_TTTEEE_lowTEB','plikHM_TTTEEE_lowTEB_post_BAO','plikHM_TTTEEE_lowTEB_post_lensing','plikHM_TTTEEE_lowTEB_post_H070p6','plikHM_TTTEEE_lowTEB_post_JLA','plikHM_TTTEEE_lowTEB_post_zre6p5','plikHM_TTTEEE_lowTEB_post_BAO_H070p6_JLA','plikHM_TTTEEE_lowTEB_post_lensing_BAO_H070p6_JLA','plikHM_TTTEEE_lowl_lensing','plikHM_TTTEEE_lowl_lensing_post_BAO','plikHM_TTTEEE_lowl_lensing_post_BAO_H070p6_JLA','plikHM_TTTEEE_lowTEB_lensing']


Models={}
Models['model']=['base','base_omegak','base_Alens','base_Alensf','base_nnu','base_mnu',\
                 'base_nrun','base_r','base_w','base_alpha1','base_Aphiphi','base_yhe',\
                 'base_mnu_Alens','base_mnu_omegak','base_mnu_w','base_nnu_mnu',\
                 'base_nnu_r','base_nnu_yhe','base_w_wa',\
                 'base_nnu_meffsterile','base_nnu_meffsterile_r']

npars=[6,7,7,7,7,7,\
       7,7,7,7,7,7,\
       8,8,8,8,\
       8,8,8,\
       8,9]                 
Models['npars']={k:v for k,v in zip(Models['model'],npars)}

pvol=[1, 0.6, 10., 10., 9.95, 5.,\
      2., 3., 4., 2., 10., 0.4,\
      50., 3., 20., 49.75,\
      29.85, 3.98, 20.0,\
      29.85, 89.55]                 
Models['priorvolume'] = {k:v for k,v in zip(Models['model'],pvol)}
                         
#----------------------------------

def avail_data_list(mm):
    l=glob.glob( '{0}/{1}/*/*_1.txt'.format(rootdir,mm) )
    l1=[x.split('_1')[0] for x in l]
    l2=[x.split('base_')[1] for x in l1]
    return l1,l2

def avail_model_list(dd,nmax=None):
    df=pd.DataFrame()
    l=glob.glob( '{0}/*/*/*_{1}_1.txt'.format(rootdir,dd) )
    df['l1']=[x.split('_1')[0] for x in l]    
    df['l2']=df['l1'].apply(lambda x:x.split('/')[1])
    df=df.sort_values('l2')
    if nmax:
        df=df.iloc[0:nmax]
    return df['l1'].values,df['l2'].values

def iscosmo_param(p,l=cosmo_params):
    return p in l

def params_info(fname):
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


#-------------------------------------

#------- model, data etc. lists ----
nchain=4
nmodel=None
ndata=1
data_list=DataSets #[0:ndata]
model_list=Models['model'] 
print('------------')
#print('data:',data_list)
#print('model:',model_list)
#print('nparams:',Models['npars'])
print('Len data, model',len(data_list),len(model_list))
print('------------')

#------ mpi load balancing ---
main_loop_list=data_list
nload=len(main_loop_list)
lpp=mpi_load_balance(mpi_size,nload)


#------------------

all_mm={}
mce_cols=['chain%s'%k for k in range(1,nchain+1)]
#mce_info_cols=['NparamsCosmo','PriorVol','NsamplesRead','NsamplesUsed']
mce_info_cols=['PriorVol','ndim','N_read','N_used']

for ipp in range(lpp[rank]): 
    ig=ipp+lpp[0:rank-1].sum()
    kk=main_loop_list[ig]
    print('*** mpi_rank, idd, loop_key',rank, ig, kk)

    kk_name='data'
    idd=ig
    dd=kk
    dd_dir=dd.split('_post_')[0]
    dd_name=dd #dd.split('plikHM_')[0]    

    path_list, name=avail_model_list(dd,nmax=nmodel)
    mce=np.zeros((len(path_list),len(mce_cols)))
    mce_info={ k:[] for k in mce_info_cols }

    vol_norm=1.0    
    for imm,mm,fname in zip(range(len(name)),name, path_list): #model loop            
        if True:#os.path.exists(fname):
            parMC=params_info(fname)
            if mm=='base':
                vol_norm=parMC['volume']
                
            prior_volume=parMC['volume']/vol_norm #Models['priorvolume'][mm] #
            ndim=parMC['ndim'] #Models['npars'][mm]  #
            #print('** model {}: ndim {} {}; volume {} {}'.format(mm,ndim,Models['npars'][mm],prior_volume,Models['priorvolume'][mm]))
            #            
            mce_info['PriorVol'].append(prior_volume)
            mce_info['ndim'].append(ndim)            
            #
            #print('***model: {},  ndim:{}, volume:{}, name={}'.format(mm,ndim,prior_volume,parMC['name']))
            #
            nc_read=''
            nc_use=''
            icc=0
            for cc in ['_%s.txt'%x for x in range(1,nchain+1)]:
                fchain=fname+cc
                e,info = MCEvidence(fchain,ndim=ndim,
                                    priorvolume=prior_volume,
                                    kmax=kmax,verbose=verbose,burnin=0,
                                    thin=False).evidence(info=True,pos_lnp=False)
                mce[imm,icc]=e[0]
                icc+=1
                nc_read=nc_read+'%s,'%info['Nsamples_read']
                nc_use=nc_use+'%s,'%info['Nsamples']
                
            mce_info['N_read'].append(nc_read)
            mce_info['N_used'].append(nc_use)
        else:
            print('*** not available: ',fname)
            mce[imm,:]=np.nan
            mce_info['N_read'].append('')
            mce_info['N_used'].append('')
            mce_info['PriorVol'].append(0)
            mce_info['ndim'].append(0)             
    
    
    if not np.all(np.isnan(mce)):
        df = pd.DataFrame(mce,index=name,columns=mce_cols)
        df_mean=df.mean(axis=1)
        df_std=df.std(axis=1)        
        #delta_df=df - df.max(axis=0)
        #
        df['Mean_lnE_k1'] =df_mean
        df['Err_lnE_k1'] = df_std/np.sqrt(nchain*1.0)
        df['delta_lnE_k1'] =df_mean-df_mean.max()
        for k in mce_info_cols:
            df[k]=mce_info[k]
            
        print('--------------- {}---------'.format(kk))
        print(tabulate(df, headers='keys', tablefmt='psql',floatfmt=".2f", numalign="left"))

        #--------- outputs ----------
        fout=fout_df.format(outdir,kk)
        print('rank-{}, writing file to {}'.format(rank,fout))
        fhandle=open(fout, 'w')
        if rank==0:
            fhandle.write('\n')
            fhandle.write('############## RootDirectory={} ########\n'.format(rootdir))
            fhandle.write('\n')

        fhandle.write('\n')                
        fhandle.write('************ {} ************'.format(kk))
        fhandle.write('\n')                
        fhandle.write(tabulate(df, headers='keys', tablefmt='psql'))
        fhandle.write('\n')
        fhandle.close()
    
        #append all tables to file
        #df_all = comm.gather(df, root=0)
        #dd_all = comm.gather(dd, root=0)        
        #if rank==0:
            # for ddi, dfi in zip(dd_all, df_all):
            #     fhandle.write('\n')                
            #     fhandle.write('************ data={} ************'.format(ddi))
            #     fhandle.write('\n')                
            #     fhandle.write(tabulate(dfi, headers='keys', tablefmt='psql'))
            #     fhandle.write('\n')
        #all_mm[dd]=df

comm.Barrier() #wait for all process to finish 

if rank==0:
    #concatnate all output files to a single file
    fmain='{}/mce_planck_fullgrid.txt'.format(outdir)
    fout_list=[fout_df.format(outdir,kk) for kk in main_loop_list]
    print('all outputs being written to ',fmain)
    with open(fmain,'w') as outfile:        
        fin=fileinput.input(fout_list)
        for line in fin:
            outfile.write(line)
        fin.close()
        
    #delete all single files
    for fname in fout_list:
       os.remove(fname)
        
# #---------------------------------------------
# #--------- gather all -----------------------
# all_mm = comm.gather(all_mm, root=0)
# if rank==0:
#     #print('after gather type(all_mm)=',type(all_mm))
#     all_mm={ k: v for d in all_mm for k, v in d.items() }
#     print ('after_gather and concat: all_mm.keys:',all_mm.keys())

#     # big_df=pd.DataFrame()
#     # for mm,df in all_mm.items():
#     #     big_df[mm]=df.loc['k1']
#     # big_df=big_df.T
#     #

#     # Save a dictionary into a pickle file.
#     fout_pkl='{0}/pla_evidence_df_dataKey_dict.pkl'.format(outdir)
#     pickle.dump(all_mm, open(fout_pkl, "wb") )
    
    ##read
    #all_mm=pickle.load( open(fout_pkl, "rb" ) )

#---------------------------------------------
