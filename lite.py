#!/usr/bin/env python

import prepfold
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize, fmin
from psr_utils import gaussian_profile
import argparse
import os

## #!/opt/pulsar/python/2.7.12/bin/python

#From gaussian_beam
def gaussian_beam(rr,amp,r0,fwhm=36.0):
    return amp *  np.exp(-2.7726 * (rr - r0)**2 / (fwhm * fwhm))

# see https://stackoverflow.com/questions/34136737/using-scipy-curve-fit-for-a-variable-number-of-parameters
def gaussprof(x, N, *args):
      a, b, c = list(args[0][:N]), list(args[0][N:2*N]), list(args[0][2*N:3*N])
      return amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def tfunc(x,*args):
    ncomps = len(args)/3
    phass = list(args[:ncomps])
    ampls = list(args[ncomps:2*ncomps])
    fwhms = list(args[2*ncomps:3*ncomps])
    y=0

    for ampl, phas, fwhm in zip(ampls, phass, fwhms):
        sigma = fwhm / 2.35482
        mean = phas % 1.0
        y+= ampl*np.exp(-(x-mean)**2/(2*sigma**2))/(sigma*(2*np.pi)**0.5)

    return y

#From read_gaussfit
def read_gaussfit(gaussfitfile, Nprofbins):

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

    # Find max value and apply corrections to amplitudes.
    neg_amp_params = list(phass) + list(-1.0*ampls) + list(fwhms)
    y_find_max = lambda x: tfunc(x,*neg_amp_params) 
    max_amp_phase = fmin(y_find_max,0.0,disp=0)    # Silently find max value (fmin with -1*func)
    ampls/=-1*y_find_max(max_amp_phase)

    prof_params = list(phass) + list(ampls) + list(fwhms)
    global template_function
    template_function = lambda x: tfunc(x, *prof_params)

    template = np.zeros(Nprofbins, dtype='d')
    for ii in range(len(ampls)):
        template += ampls[ii]*gaussian_profile(Nprofbins, phass[ii], fwhms[ii])

    return template


def scale_template(x,a):
    return a*template_function(x)


# Using supplied profile model and threshold, determine on/off bins.
def on_off(fname,nbin,threshold=0.05):

    template = read_gaussfit(fname,nbin)
    template = template
    on_bins = np.where(template>threshold)[0]
    off_bins = np.where(template<threshold)[0]

    return on_bins, off_bins


def amps_errs(profs,temp_fname):
    nsub    = profs.shape[0]
    nbin    = profs.shape[1]

    amps = []
    errs = []
    
    phase = np.arange(nbin)/np.float(nbin)
    template = read_gaussfit(temp_fname,nbin)
    on_inds, off_inds = on_off(temp_fname,nbin)
    #print on_inds,off_inds
    #print "%s bins in ON window." % (len(on_inds))
    #plt.plot(template, 'k', template_function(phase), '--r')
    #plt.show()

    # Off-pulse analysis:
    all_off_mean = np.mean(profs[:,off_inds])
    all_off_std = np.std(profs[:,off_inds])
    #folded_prof = np.sum(profs,axis=0)
    #folded_off_mean = np.mean(folded_prof[off_inds])
    #folded_off_std = np.std(folded_prof[off_inds])
    #print all_off_std, folded_off_std, all_off_mean, folded_off_mean

    profs -= all_off_mean
    profs /= all_off_std

    #bg_subtract = []
    #std_div = []
    for i,p in enumerate(profs):
        off_pulse_ints = p[off_inds]
        off_mean, off_std = np.mean(off_pulse_ints), np.std(off_pulse_ints)
        #print off_mean,off_std,max(p)
        p -= off_mean
        #bg_subtract.append(off_mean)
        #std_div.append(off_std)
        #p /= all_off_std
        # Beam profiles look MUCH better dividing by full STD rather than STD of each profile's noise.
        # Subtracting by all_off_mean shows pronounced (and often periodic) wiggles in the beam profile??!
        # ...therefore, it looks more reliable to subtract "per profile" baselines, but I'll need to think more about this.
        #print ""
        #print off_mean/all_off_mean, off_std/all_off_rms
        a_guess = max(p)-min(p)
        popt, pcov = curve_fit(scale_template, phase, p, p0=[a_guess])

        #increment = (i+0.5)/15.0
        #plt.plot(phase,p+increment,'k')
        #plt.plot(phase,scale_template(phase,popt[0])+increment,'--r')

        mult_factor = 1.0
        if np.abs(off_mean) > 1.0: mult_factor = off_mean**2
        amps.append(popt[0])
        errs.append(np.sqrt(np.diag(pcov))[0]*mult_factor)
        # ARE ERRORS STILL SCALING AS EXPECTED?
        #print max(p),1.15*np.std(p[off_inds])/np.sqrt(float(len(off_inds))), errs[-1]

    ##plt.plot(phase,scale_template(phase,1.0),'--g')

    # WEIRD STUFF (background subtraction)
    #plt.plot(bg_subtract/max(bg_subtract),':b',std_div/max(std_div),'--r')
    #plt.show()
    return np.asarray(amps), np.asarray(errs)

# NEED TO INCORPORATE ERRORS!!!
def fit_1d_bp(offs,amps,errs):
    amp_guess = np.max(amps)-np.min(amps)
    loc_guess = offs[np.argmax(amps)]
    errs = np.asarray(errs)
    amps = np.asarray(amps)
    popt,pcov = curve_fit(gaussian_beam,offs[amps>0],amps[amps>0],p0=[amp_guess,loc_guess],sigma=errs[amps>0],absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr 

def get_bp(scan_fname,temp_fname,direction,dr=0.1333):

    pf = prepfold.pfd(scan_fname)
    pf.dedisperse()

    # BEST WAY TO DO THIS?? 
    nsub    = pf.npart
    nchan   = 1
    nbin    = pf.proflen
 
    profs = pf.combine_profs(nsub,nchan).squeeze()

    # Calculate position offsets.
    mid_time = (pf.mid_secs[0]+pf.mid_secs[-1])/2.0
    times = np.linspace(pf.mid_secs[0],pf.mid_secs[-1],nsub) - mid_time
    offsets_arcmin = times*dr

    # Is this necessary? Uses the template to determine, given some threshold.
    #on_inds, off_inds = on_off(model_fname,nbin,test=0)
    ii,ee = amps_errs(profs,temp_fname)
    

    # ***  Probably need a cos(dec) correction for offsets_arcmin...!  ***
    # 1=Dec, 0=RA
    if direction == 1:
        yy = offsets_arcmin
        xx = np.zeros(len(yy))
        offs = yy

    elif direction == 0:
        xx = offsets_arcmin
        yy = np.zeros(len(xx))
        offs = xx

    # Fit beam profile, print/plot results
    bp_vals, bp_errs = fit_1d_bp(offs,ii,ee)
    #plt.errorbar(offs,ii,yerr=ee,fmt='o',capsize=3)
    #plt.plot(offs,gaussian_beam(offs,bp_vals[0],bp_vals[1]),'--r')
    print bp_vals,bp_errs
    resids = ii-gaussian_beam(offs,bp_vals[0],bp_vals[1])
    R_std = np.std(resids[ii>0])
    
    #print "Offset    Err:"
    #for oset, err, amp in zip(offs,ee,ii): print oset,err, amp
    
    plt.figure(0)
    plt.errorbar(offs,ii/R_std,yerr=ee/R_std,fmt='o',capsize=3)
    plt.plot(offs,gaussian_beam(offs,bp_vals[0],bp_vals[1])/R_std,'--r')
    plt.figure(1)
    plt.errorbar(offs,resids, yerr=ee/R_std,fmt='o',capsize=3)
    plt.plot(offs,[0]*len(offs),'--r')
    plt.show()


    return np.array(xx), np.array(yy), np.array(ii), np.array(ee)


class loc_info:

    def __init__(self,filename):

        if os.path.isfile(filename):
            f = open(filename, 'r')

        # Probably a cleaner way -- plus, makes some assumptions about directory structure.
        if '/' in filename:
            x = filename.split('/')
            while '' in x: x.remove('')
            self.loc_file = x.pop()
            self.path = '/'.join(x)
        else:
            self.loc_file = filename
            self.path = './'

        self.scan_files = []
        self.temp_files = []
        self.directions = []

        for line in f:
            if line.strip()[0] == '#': continue
            
            # Eventually, make this more generic for N scans in either RA/DEC directions.
            a = line.split('=')
            if a[0].count('Scan'):
                self.scan_files.append(self.path+'/'+str(a[1]).strip())
            if a[0].count('Template'):
                self.temp_files.append(self.path+'/'+str(a[1]).strip())
            # 0 = RA, 1 = Dec
            if a[0].count('Direction'):
                self.directions.append(int(a[1]))
            if a[0].count('Drift Rate'):
                self.driftrate = float(a[1])
            else:
                continue

        f.close()

        self.x = np.empty(0)
        self.y = np.empty(0)
        self.i = np.empty(0)
        self.di = np.empty(0)

        # Now it's time to get relevant offset/intensity info.
        for ss,tt,dd in zip(self.scan_files, self.temp_files, self.directions):
            ox,oy,oi,odi = get_bp(ss,tt,dd,dr=self.driftrate) 
            self.x = np.append(self.x,ox)
            self.y = np.append(self.y,oy)
            self.i = np.append(self.i,oi)
            self.di = np.append(self.di,odi)

        #  ...and eventually do full, 2D fit.


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('loc_file')
    args = parser.parse_args()

    x = loc_info(args.loc_file) 
    #print x.x
    #print x.y
    #print x.i
    #print x.di
