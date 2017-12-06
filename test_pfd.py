#!/opt/pulsar/python/2.7.12/bin/python
import prepfold
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from psr_utils import gaussian_profile
from sys import argv

#From gaussian_beam
def gaussian_beam(rr,amp,offset,fwhm=36.0,obsfreq=350.0):
  return amp *  np.exp(-2.7726 * (rr - offset)**2 / (fwhm * fwhm))

def gaussprof(x, N, *args):
# see https://stackoverflow.com/questions/34136737/using-scipy-curve-fit-for-a-variable-number-of-parameters
    a, b, c = list(args[0][:N]), list(args[0][N:2*N]), list(args[0][2*N:3*N])
    return amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#From read_gaussfit
def read_gaussfit(gaussfitfile, proflen):
    """
    read_gaussfitfile(gaussfitfile, proflen):
        Read a Gaussian-fit file as created by the output of pygaussfit.py.
            The input parameters are the name of the file and the number of
            bins to include in the resulting template file.  A numpy array
            of that length is returned.
    """
    phass = []
    ampls = []
    fwhms = []
    for line in open(gaussfitfile):
        if line.lstrip().startswith("const"):
            const = float(line.split()[2])
        if line.lstrip().startswith("phas"):
            phass.append(float(line.split()[2]))
        if line.lstrip().startswith("ampl"):
            ampls.append(float(line.split()[2]))
        if line.lstrip().startswith("fwhm"):
            fwhms.append(float(line.split()[2]))
    if not (len(phass) == len(ampls) == len(fwhms)):
        print "Number of phases, amplitudes, and FWHMs are not the same in '%s'!"%gaussfitfile
        return 0.0
    phass = np.asarray(phass)
    ampls = np.asarray(ampls)
    fwhms = np.asarray(fwhms)

    def tfunc(x,*args):
	#print type(args), len(args)
        ncomps = len(args)/3
        phass = list(args[:ncomps])
        ampls = list(args[ncomps:2*ncomps])
        fwhms = list(args[2*ncomps:3*ncomps])
	#const = args[-1]
        #y=const
	y=0
	for ampl, phas, fwhm in zip(ampls, phass, fwhms):
            sigma = fwhm / 2.35482
            mean = phas % 1.0
            y+= ampl*np.exp(-(x-mean)**2/(2*sigma**2))/(sigma*(2*np.pi)**0.5)
        
        return y
    
    prof_params = list(phass) + list(ampls) + list(fwhms) + [const]
    global template_function
    template_function = lambda x: tfunc(x, *prof_params)

    template = np.zeros(proflen, dtype='d')
    for ii in range(len(ampls)):
        template += ampls[ii]*gaussian_profile(proflen, phass[ii], fwhms[ii])
    return template

'''def template_func(gaussfitfile):
	phass = []
	ampls = []
	fwhms = []
	for line in open(gaussfitfile):
		if line.lstrip().startswith("phas"):
			phass.append(float(line.split()[2]))
		if line.lstrip().startswith("ampl"):
			ampls.append(float(line.split()[2]))
		if line.lstrip().startswith("fwhm"):
			fwhms.append(float(line.split()[2]))
	if not (len(phass) == len(ampls) == len(fwhms)):
		print "Number of phases, amplitudes, and FWHMs are not the same in '%s'!"%gaussfitfile
		return 0.0
	#phass = np.asarray(phass)
	#ampls = np.asarray(ampls)
	#fwhms = np.asarray(fwhms)
	template = np.zeros(proflen, dtype='d')
	ncomps = len(ampls)
	def tfunc(x,*args):
		ncomps = len(args[0])/3
		phass = list(args[0][:ncomps])
		ampls = list(args[0][ncomps:2*ncomps])
		fwhms = list(args[0][2*ncomps:3*ncomps])
		y=0
		for ampl, phas, fwhm in zip(ampls, phass, fwhms):
			sigma = fwhm / 2.35482
			mean = phas % 1.0
			y+= ampl*np.exp(-(x-mean)**2/(2*sigma**2))
		
		return y
	
	prof_params = phass+ampls+fwhms
	global template_function
	template_function = lambda x: tfunc(x, *prof_params)
	
	def scaled_template(x,c):
		
		
	funcs = []
	for ii in range(len(ampls)):
		lambda x, a, m, s: gauss(x,a,m,s)
		funcs.append(lambda x, a, m, s: gauss(x,a,m,s))
	
	return template'''

def scale_template(x,a):
	return a*template_function(x)


# Beam profile object
class bpo:
  def __init__(self,offsets,intensities):
    self.offs = offsets
    self.bp = intensities

    self.fname = None
    self.times = None
    self.drift_rate = None
 
  def fit_bp(self):
    popt,pcov = curve_fit(gaussian_beam,self.offs,self.bp,p0=[1.0,0.0])
    fit_data = gaussian_beam(self.offs,*popt)
    residuals = self.bp-fit_data
    
    #print ""
    #print "COVARIANCE MATRIX:"
    #print pcov
    #print ""
    perr = np.sqrt(np.diag(pcov))
    self.amp = popt[0]
    self.amperr = perr[0]
    self.off = popt[1]
    self.offerr = perr[1]

    # print pertinent info:
    #print "** Beam profile fit uses %i SNR values **" % (len(self.offs))
    #print "       OFFSET  = %.1f +/- %.1f" % (self.off,self.offerr)
    #print "       AMP     = %.1f +/- %.1f" % (self.amp,self.amperr)
    #print "       BP PEAK = (BP peak in seconds from start of obs?)" 
    #print "#Nsubints, bestfit offset, bestfit offset err"
    print len(self.offs), self.off, self.offerr


  # Probably want a general plotter to do RA/DEC + composite.
  def plot_fit(self):
    plt.plot(self.offs,self.bp)
    plt.plot(self.offs,gaussian_beam(self.offs,self.amp,self.off),c='red',ls='--',lw=2)

    plt.title(self.fname.split("/")[-1])
    plt.xlabel('Offset (arcmin)')
    plt.ylabel('Signal-to-Noise Ratio')
    plt.show()


# Using supplied profile model and threshold, determine on/off bins.
def on_off(fname,nbin,threshold=0.05,test=0):
  template = read_gaussfit(fname,nbin)
  template = template/np.max(template)
  on_bins = np.where(template>threshold)
  off_bins = np.where(template<threshold)

  if test:
    on_region = np.zeros(len(template))
    on_region[on_bins] = 0.25 
    plt.plot(template)
    plt.plot(on_region,c='r')
    plt.show()
    exit()

  return on_bins, off_bins


def get_snrs(profs,model_fname, mode = "snr"):
	nsub    = profs.shape[0]
	nchan   = 1
	nbin    = profs.shape[-1]
	
	if mode == "snr":
		on_inds, off_inds = on_off(model_fname,nbin,test=0)
	
		#print ""
		#print "ON-PULSE BINS:"
		#print on_inds
		#profs = pf.combine_profs(nsub,nchan)		 # [isub:ichan:ibin]
		snrs = []
		sub_test = []
		for isub in xrange(nsub):
			for ichan in xrange(nchan):
				profile	= profs[isub,ichan] 
				mean_off = np.mean(profile[off_inds])
				std_off	= np.std(profile[off_inds])
				snr = np.sum(profile-mean_off)/std_off	 # lk+05 (no arbitrary width term)
				snrs.append(snr)
				#print isub, mean_off, snr

		snrs =	np.array(snrs)
		return snrs
		
	elif mode == "global_max":
		on_inds, off_inds = on_off(model_fname,nbin,test=0)
		profs = profs.squeeze()
		for i in range(nsub):
			profile = profs[i,:]
			prof_baseline = np.average(profile[off_inds])
			profs[i,:] -= prof_baseline
				
		snrs = profs.max(axis=1)
		mins = profs.min(axis=1)
		return snrs
	
	elif mode == "profile_max":
		on_inds, off_inds = on_off(model_fname,nbin,test=0)
		profs = profs.squeeze()
		for i in range(nsub):
			profile = profs[i,:]
			prof_baseline = np.average(profile[off_inds])
			profs[i,:] -= prof_baseline

		snrs = profs[:,on_inds].squeeze().max(axis=1)
		'''print on_inds
		for i in range(nsub):
			plt.plot(profs[i,:])
			plt.plot(list(on_inds[0]),[snrs[i]]*len(on_inds[0]))
			plt.show()		
		exit(0)'''
		return snrs
	elif mode == "subint_fit":
		on_inds, off_inds = on_off(model_fname,nbin,test=0)
		profs = profs.squeeze()
		snrs = []
		phases = np.linspace(0,1-1.0/nbin,nbin)
		for i in range(nsub):
			#if i>15:
				profile = profs[i,:]
				prof_baseline = np.average(profile[off_inds])
				profile-=prof_baseline
				a_guess = max(profile)-min(profile)
				popt, pcov = curve_fit(scale_template, phases, profile, p0=[a_guess])
				snrs.append(popt[0])
			

		return snrs


def get_bpo(pfd_fname,model_fname,dr=0.666,mode="snr",nsubints=None):

  pf = prepfold.pfd(pfd_fname)
  pf.dedisperse()

  # BEST WAY TO DO THIS?? 
  nsub    = pf.npart
  nchan   = 1
  nbin    = pf.proflen
  
  if nsubints and nsub>nsubints:
    if nsub%nsubints:
      newsubints = float(nsub)/(int(nsub)/int(nsubints))
    else:
       newsubints = nsubints
    
    profs = pf.combine_profs(newsubints,nchan)
  elif nsub<nsubints:
    print "Cannot scrunch to %d subints since original file has %d subints." % (nsubints,nsub)
  else:
    newsubints = nsub
  
  profs = pf.combine_profs(newsubints,nchan)

  print pf.mid_secs[0],pf.mid_secs[-1]
  exit(0)
  # Calculate position offsets. First method doesn't work well. Why?
  mid_time = (pf.mid_secs[0]+pf.mid_secs[-1])/2.0
  times = np.linspace(pf.mid_secs[0],pf.mid_secs[-1],newsubints) - mid_time
  offsets_arcmin = times*dr#*(nsub/newsubints)  # dr is in arcmin/second

  #pdata = pf.combine_profs(1,1).flatten()
  on_inds, off_inds = on_off(model_fname,nbin,test=0)

  '''print ""
  print "ON-PULSE BINS:"
  print on_inds
  profs = pf.combine_profs(nsub,nchan)     # [isub:ichan:ibin]
  snrs = []
  sub_test = []
  for isub in xrange(nsub):
    for ichan in xrange(nchan):
      profile  = profs[isub,ichan] 
      mean_off = np.mean(profile[off_inds])
      std_off  = np.std(profile[off_inds])
      snr = np.sum(profile-mean_off)/std_off   # lk+05 (no arbitrary width term)
      snrs.append(snr)
      #print isub, mean_off, snr

  snrs =  np.array(snrs)'''

  snrs=get_snrs(profs,model_fname,mode)

  # Get BPO ready to return...
  x = bpo(offsets_arcmin,snrs)
  x.fname = pfd_fname
  x.times = times  #_0
  x.drift_rate = dr
  return x 


def prof_test(model_fname,nbin,threshold=0.05):
  on_off(model_fname,nbin,threshold=threshold,test=1)


if __name__ == "__main__":
	nsub=None
	make_plot = True
	args = argv[1:]
	for i,arg in enumerate(args):
		if arg == "-pfd":
			pfd_fname = args[i+1]
		elif arg in ["-template", "-t"]:
			template_fname = args[i+1]
		elif arg in ["-mode","-m"]:
			mode = args[i+1]
		elif arg == "-nsub":
			nsub = int(args[i+1])
		elif arg == "-noplot":
			make_plot = False
	
	driftrate = 36.0*3/(12*60.0)
	x = get_bpo(pfd_fname, template_fname, dr=driftrate, mode=mode, nsubints=nsub)
	#x = get_bpo('/lakitu/data/swiggum/DRIFT/presto_fit/2038/guppi_57791_2038-36_0003_0001_3.28ms_Cand.pfd','/lakitu/data/swiggum/DRIFT/presto_fit/2038/2038.dec.gaussians',dr=driftrate)
	#prof_test('1921.dec.gaussians',64)
	#driftrate = 0.1333
	#x = get_bpo('/lakitu/data/swiggum/DRIFT/presto_fit/1919/guppi_56902_B1919+21_0007_0001_PSR_1921+2153.pfd','/lakitu/data/swiggum/DRIFT/presto_fit/1919/1921.dec.gaussians',dr=driftrate)
	#x = get_bpo('/lakitu/data/swiggum/DRIFT/presto_fit/1154/guppi_57791_1154-19_0046_0001_11.12ms_Cand.pfd','/lakitu/data/swiggum/DRIFT/presto_fit/1154/1154.ra.gaussians',dr=driftrate)
	x.fit_bp()
	if make_plot: x.plot_fit()
