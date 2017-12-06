#!/usr/bin/python
import prepfold
import numpy as np
import matplotlib.pyplot as plt
from read_gaussfit import *
from scipy.optimize import curve_fit
from gaussian_beam import *

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
    print ""
    print "COVARIANCE MATRIX:"
    print pcov
    print ""
    perr = np.sqrt(np.diag(pcov))
    self.amp = popt[0]
    self.amperr = perr[0]
    self.off = popt[1]
    self.offerr = perr[1]

    # print pertinent info:
    print "** Beam profile fit uses %i SNR values **" % (len(self.offs))
    print "       OFFSET  = %.1f +/- %.1f" % (self.off,self.offerr)
    print "       AMP     = %.1f +/- %.1f" % (self.amp,self.amperr)
    print "       BP PEAK = (BP peak in seconds from start of obs?)" 

  # Probably want a general plotter to do RA/DEC + composite.
  def plot_fit(self):
    plt.plot(self.offs,self.bp)
    plt.plot(self.offs,gaussian_beam(self.offs,self.amp,self.off),c='red',ls='--',lw=2)

    plt.title(self.fname)
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


def get_bpo(pfd_fname,model_fname,dr=0.666):

  pf = prepfold.pfd(pfd_fname)
  pf.dedisperse()

  # BEST WAY TO DO THIS?? 
  nsub    = pf.npart
  nchan   = 1
  nbin    = pf.proflen
  print nsub

  # Calculate position offsets. First method doesn't work well. Why?
  mid_time = (pf.mid_secs[0]+pf.mid_secs[-1])/2.0
  times = np.array(pf.mid_secs)-mid_time
  offsets_arcmin = times*dr   # dr is in arcmin/second

  pdata = pf.combine_profs(1,1).flatten()
  on_inds, off_inds = on_off(model_fname,nbin,test=0)

  print ""
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

  snrs =  np.array(snrs)

  # Get BPO ready to return...
  x = bpo(offsets_arcmin,snrs)
  x.fname = pfd_fname
  x.times = times  #_0
  x.drift_rate = dr
  return x 


def prof_test(model_fname,nbin,threshold=0.05):
  on_off(model_fname,nbin,threshold=threshold,test=1)


if __name__ == "__main__":

  driftrate = 36.0*3/(12*60.0)
  x = get_bpo('./2038/guppi_57791_2038-36_0002_0001_3.28ms_Cand.pfd','./2038/2038.ra.gaussians',dr=driftrate)
  #x = get_bpo('./2038/guppi_57791_2038-36_0003_0001_3.28ms_Cand.pfd','./2038/2038.dec.gaussians',dr=driftrate)
  #prof_test('1921.dec.gaussians',64)
  #driftrate = 0.1333
  #x = get_bpo('./1919/guppi_56902_B1919+21_0007_0001_PSR_1921+2153.pfd','./1919/1921.dec.gaussians',dr=driftrate)
  #x = get_bpo('./1154/guppi_57791_1154-19_0046_0001_11.12ms_Cand.pfd','./1154/1154.ra.gaussians',dr=driftrate)
  x.fit_bp()
  x.plot_fit()

