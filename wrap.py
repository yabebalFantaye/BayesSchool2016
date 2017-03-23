from __future__ import print_function
import IPython
#
import os, sys, math
import time
import numpy as np

#pretty plots if seaborn is installed
try: 
    import seaborn as sns
    sns.set(style='ticks', palette='Set2',font_scale=1.5)
    #sns.set() 
except:
    pass

#====================================
#      Getdist wrapper
#====================================
try:
    from getdist import plots, MCSamples
    import getdist as gd
    class samples2gdist(object):
        #Ref:
        #http://getdist.readthedocs.io/en/latest/plot_gallery.html

        def __init__(self,chain,lnprob,trueval=None,
                    names=None,labels=None,px='x'):
            #Get the getdist MCSamples objects for the samples, specifying same parameter
            #names and labels; if not specified weights are assumed to all be unity

            ndim=chain.shape[-1]

            if names is None:
                names = ["%s%s"%('p',i) for i in range(ndim)]
            if labels is None:
                labels =  ["%s_%s"%(px,i) for i in range(ndim)]

            self.names=names
            self.labels=labels
            self.trueval=trueval
            self.samples = gd.MCSamples(samples=chain,loglikes=lnprob,
                                 names = names, labels = labels)

        def triangle(self,**kwargs):
            #Triangle plot
            g = gd.plots.getSubplotPlotter()
            g.triangle_plot(self.samples, filled=True,**kwargs)

        def corner(self,figsize=(15,15),**kwargs):
            #let's trangle plot chain samples
            try:
                import corner
                fig = corner.corner(self.samples.samples, labels=['$%s$'%x for x in self.labels],
                              truths=self.trueval, quantiles=[0.16, 0.5, 0.84], 
                            show_titles=True, labels_args={"fontsize": 10})
                fig.set_size_inches(figsize)        
            except:
                print('the package corner is not installed. Using getdist triangle_plot instead')
                self.triangle()

        def plot_1d(self,l,**kwargs):
            #1D marginalized plot
            g = gd.plots.getSinglePlotter(width_inch=4)        
            g.plot_1d(self.samples, l,**kwargs)

        def plot_2d(self,l,**kwargs):
            #Customized 2D filled comparison plot
            g = gd.plots.getSinglePlotter(width_inch=6, ratio=3 / 5.)       
            g.plot_1d(self.samples, l,**kwargs)      

        def plot_3d(self,llist):
            #2D scatter (3D) plot
            g = gd.plots.getSinglePlotter(width_inch=5)
            g.plot_3d(self.samples, llist)   

        def save_to_file(self,path=None,dname='chain',froot='test'):
            #Save to file
            import tempfile, os
            if path is None:
                path=tempfile.gettempdir()
            tempdir = os.path.join(path,dname)
            if not os.path.exists(tempdir): os.makedirs(tempdir)
            rootname = os.path.join(tempdir, froot)
            self.samples.saveAsText(rootname)     

        def load_from_file(rootname):
            #Load from file
            self.samples=[]
            for f in rootname:
                self.samples.append(gd.loadMCSamples(rootname))

        def info(self): 
            #these are just to show getdist functionalities
            print(self.samples.PCA(['x1','x2']))
            print(self.samples.getTable().tableTex())
            print(self.samples.getInlineLatex('x1',limit=1))
except:    
    print('getdist is not installed. You can not use the wrapper: samples2gdist')                      
#====================================
#      Emcee wrapper
#====================================
#
try:
    import emcee

    class make_emcee_chain(object):
        # A wrapper to the emcee MCMC sampler
        #
        def __init__(self,model,nwalkers=500,nburn=300,arg={}):

            #check if model is string or not
            if isinstance(model,str):
                print('name of model: ',model)
                XClass = getattr(sys.modules[__name__], model)
            else:            
                XClass=model        

            #check if XClass is instance or not
            if hasattr(XClass, '__class__'): 
                print('instance of a model class is passed')
                self.model=XClass #it is instance 
            else:
                print('class variable is passed .. instantiating class')
                self.model=XClass(*arg)

            self.ndim=self.model.ndim

            #init emcee sampler
            self.nwalkers=nwalkers
            self.emcee_sampler = emcee.EnsembleSampler(self.nwalkers, 
                                                 self.model.ndim, 
                                                 self.model.lnprob)   

            # burnin phase
            pos0=self.model.pos(self.nwalkers)
            pos, prob, state  = self.emcee_sampler.run_mcmc(pos0, nburn)

            #save emcee state
            self.prob=prob
            self.pos=pos
            self.state=state

            #discard burnin chain 
            self.samples = self.emcee_sampler.flatchain        
            self.emcee_sampler.reset()

        def mcmc(self,nmcmc=2000,**kwargs):
            # perform MCMC - no resetting 
            # size of the chain increases in time
            time0 = time.time()
            #
            #pos=None makes the chain start from previous state of sampler
            self.pos, self.prob, self.state  = self.emcee_sampler.run_mcmc(self.pos,nmcmc,**kwargs)
            self.samples = self.emcee_sampler.flatchain    
            self.lnp = self.emcee_sampler.flatlnprobability
            #
            time1=time.time()
            #
            print('emcee total time spent: ',time1-time0)        
            print('samples shape: ',self.samples.shape)  

            return self.samples,self.lnp

        def Sampler(self,nsamples=2000):
            # perform MCMC and return exactly nsamples
            # reset sampler so that chains don't grow
            #
            N=(nsamples+self.nwalkers-1)/self.nwalkers #ceil to next integer
            print('emcee: nsamples, nmcmc: ',nsamples,N*self.nwalkers)
            #
            #pos=None makes the chain start from previous state of sampler
            self.pos, self.prob, self.state  = self.emcee_sampler.run_mcmc(self.pos,N)
            self.samples = self.emcee_sampler.flatchain    
            self.lnp = self.emcee_sampler.flatlnprobability
            self.emcee_sampler.reset()

            return self.samples[0:nsamples,:],self.lnp[0:nsamples]

        def vis(self,chain=None,figsize=(10,10),**kwargs):
            # Visualize the chains

            if chain is None:
                chain=self.samples

            fig = corner.corner(chain, labels=self.model.label, 
                                       truths=self.model.p,
                                       **kwargs)            

            fig.set_size_inches(figsize)  

        def info(self):
            print("Example using emcee sampling") 
            print('nwalkers=',self.walkers)
            try:
                self.model.info()
            except:
                pass
            print()  
            
except:    
    print('emcee is not installed. You can not use the wrapper: make_emcee_chain')    
    

#======================================
#   Thin samples 
#======================================
def thin_samples(samples,lnp,w,nminwin=5,nthin=None):
    print('Thinning samples ..')
    gd_mc=samples2gdist(samples,lnp,weight=w,px='m')
    if nthin is None:
        ncorr=max(1,int(gd_mc.samples.getCorrelationLength(nminwin)))
    else:
        ncorr=nthin
        
    print('Samples are thinned by the correlation length of ',ncorr)
    
    gd_mc.samples.thin(ncorr)
    thin_samples=gd_mc.samples.samples
    thin_lnp=-gd_mc.samples.loglikes
    thin_w=gd_mc.samples.weights
    
    print('Chain length before and after thinning: ',len(lnp),len(thin_lnp))
    
    return thin_samples,thin_lnp,w

    
    
#====================================
#      corner (triangle plot) use example 
#====================================
#    
try:
    import corner 
    def corner_test():
        # Set up the parameters of the problem.
        ndim, nsamples = 3, 50000

        # Generate some fake data.
        np.random.seed(42)
        data1 = np.random.randn(ndim * 4 * nsamples // 5).reshape([4 * nsamples // 5, ndim])
        data2 = (4*np.random.rand(ndim)[None, :] + np.random.randn(ndim * nsamples // 5).reshape([nsamples // 5, ndim]))
        data = np.vstack([data1, data2])

        # Plot it.
        figure = corner.corner(data, labels=[r"$x$", r"$y$", r"$\log \alpha$", r"$\Gamma \, [\mathrm{parsec}]$"],
                               quantiles=[0.16, 0.5, 0.84],
                               show_titles=True, title_kwargs={"fontsize": 12})    
except:
    print('the corner package is nice chain visualiser. Install it or comment out the lines below.')
#

import time
class Timer(object):
    #Ref: https://www.huyng.com/posts/python-performance-analysis
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: %f ms' % self.msecs)
