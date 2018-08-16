#!/usr/bin/env python

from prepfold import pfd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize, fmin
from scipy.stats import iqr
from psr_utils import gaussian_profile
import argparse
import os
import matplotlib.gridspec as gridspec
import itertools as it

from astropy.coordinates import SkyCoord,ICRS,Angle,Latitude,Longitude
import astropy.units as u


def dlon_dlat(coord1,coord2):
    sep = coord1.separation(coord2).arcmin
    pa  = coord1.position_angle(coord2).deg

    dlon = sep * np.sin(pa*np.pi/180.0)
    dlat = sep * np.cos(pa*np.pi/180.0)

    return dlon,dlat


#From gaussian_beam
def gaussian_beam(rr,amp,r0,fwhm=36.0):
    return amp *  np.exp(-2.7726 * (rr - r0)**2 / (fwhm * fwhm))

def gaussian_beam_2d(P,amp,x0,y0,fwhm=36.0):
    xx,yy = P
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2))) 
    return amp * np.exp((-(xx-x0)**2 - (yy-y0)**2) / (2 * sigma**2))

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
    max_amp_phase = fmin(y_find_max,0.5,disp=0)    # Silently find max value (fmin with -1*func)
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

# Formatting positions for output
def coord_string(c_str,c_type='hms'):

    if c_type == 'hms':
        return c_str.to_string(unit=u.hour,sep=':')
    elif c_type == 'dms':
        if Latitude(c_str) > 0.0: sign = '+'
        elif Latitude(c_str) < 0.0: sign = ''
        return sign+c_str.to_string(unit=u.degree,sep=':')
    else:
        print "coord_string s_type=%s not recognized" % (s_type)
        exit()


def fit_1d_bp(offs,amps,errs):
    amp_guess = np.max(amps)-np.min(amps)
    loc_guess = offs[np.argmax(amps)]
    popt,pcov = curve_fit(gaussian_beam,offs,amps,p0=[amp_guess,loc_guess],sigma=errs,absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr 


class scan:

    def __init__(self,pfd_file,temp_file,direction,rate,threshold,order):

        pf = pfd(pfd_file)
        self.temp_file = temp_file
        self.rate = rate
        self.direction = direction
        self.thresh = threshold
        self.order = order
        self.length = pf.T

        pf.dedisperse()

        self.nsub    = pf.npart
        self.nbin    = pf.proflen
        self.phase   = np.arange(self.nbin)/np.float(self.nbin)

        self.template = read_gaussfit(self.temp_file,self.nbin)
        self.on = np.where(self.template>self.thresh)[0]
        self.off = np.where(self.template<self.thresh)[0]

        # Axis 0 = subints, axis 1 = channels, axis 2 = phase bins
        bandpass_off = np.median(pf.profs[:,:,self.off],axis=(0,2),keepdims=True)
        self.subbed_bandpass = bandpass_off.squeeze()
        temp = pf.profs - bandpass_off
        self.raw_profs = np.mean(temp,axis=1).squeeze()
        # Bandpass subtraction here makes raw_profs a bit of a misnomer (and now, = sub_profs)

        # Subtract the median from each profile (slick version!)
        self.sub_profs = self.raw_profs #- np.median(self.raw_profs[:,self.off],axis=1,keepdims=True)
        self.subbed_profile = np.sum(self.sub_profs,axis=0).squeeze()

        # Note: pf.mid_secs represents the time associated with the center of each subint.
        self.times = pf.mid_secs
        self.offs = self.times * self.rate    # Position offset from start position (pf.rastr/pf.decstr)

        # Scott: pfd header positions correspond to the telescope position on the leading edge of the first subint
        # This may disagree w/ expected starting position since there's a lag (~5 sec?) between the start
        # of a scan and when GUPPI starts writing data.
        self.actual_start_ra  = pf.rastr
        self.actual_start_dec = pf.decstr 
        pos_str = pf.rastr+' '+pf.decstr
        self.actual_start_coord = SkyCoord(pos_str, frame=ICRS, unit=(u.hourangle,u.deg), obstime="J2015.0")
        
        # Remove DC/scale by STDEV, subtract polynomials, remove DC/scale by STDEV.
        self.cleaned_profs = self.clean_profs(self.sub_profs)
        self.folded_profile = np.sum(self.cleaned_profs,axis=0).squeeze()
        self.amps_errs(self.cleaned_profs)
   
        xy = self.get_xy()
        self.xx = xy[0]
        self.yy = xy[1]

        self.efac = 1.0
        self.scale_errors()

        #"""
        # Bootstrap?
        self.avals = []    # Amplitude values
        self.ovals = []    # Offset values
        self.n_bootstrap_trials = 10000
        
        # For RFI/nulling test, remove N points from consideration:
        #n_removed = 10 
        #pass_inds = np.random.randint(0,high=self.nsub,size=self.nsub-n_removed)

        for k in range(self.n_bootstrap_trials):
            boot_inds = np.random.randint(0,high=self.nsub,size=self.nsub)
            bp_vals, bp_errs = fit_1d_bp(self.offs[boot_inds],self.ii[boot_inds],self.ee[boot_inds])
            self.avals.append(bp_vals[0])
            self.ovals.append(bp_vals[1])

        self.bp_vals = [np.mean(self.avals),np.mean(self.ovals)]
        self.bp_errs = [np.std(self.avals),np.std(self.ovals)]
        #"""

        #self.bp_vals,self.bp_errs = fit_1d_bp(self.offs,self.ii,self.ee)

        # Use 1D fit results as initial guesses for full, 2D fit.
        self.amp_guess = self.bp_vals[0]
        self.off_guess = self.bp_vals[1]
        #print '...end bootstrap...',datetime.now()


        # USE 1D FITS TO CALCULATE BEST RA/DEC VALUES
        # Direction = 1 (dec); no correction necessary!
        if self.direction:
            xxx = self.actual_start_coord.dec+Angle(self.off_guess,u.arcmin)
            dd  = coord_string(xxx,c_type='dms') 
            #print "Best Dec = %s" % (dd)
            self.best = dd 

        # Direction = 0 (ra); cos(dec) factor necessary!
        else:
            cos_dec_factor = 1.0/np.cos(self.actual_start_coord.dec)
            xxx = self.actual_start_coord.ra+Angle(self.off_guess * cos_dec_factor * u.arcmin)
            aa  = coord_string(xxx,c_type='hms') 
            #print "Best RA = %s" % (aa)
            self.best = aa

    def get_recovered_skycoord(self,position_string):
        coord_list = ['aa','dd']
        coord_list[self.direction] = self.best
        coord_list[(self.direction+1)%2] = position_string    # Could be RA/Dec!
        return SkyCoord(coord_list[0],coord_list[1], frame=ICRS, unit=(u.hourangle,u.deg), obstime="J2015.0")

    def get_dir_string(self):
        dir_strings = ['R.A.','Dec.']
        return dir_strings[self.direction]

    def scale_errors(self):
        bp_vals, bp_errs = fit_1d_bp(self.offs,self.ii,self.ee)
        chisq = np.sum((self.ii-gaussian_beam(self.offs,bp_vals[0],bp_vals[1]))**2/self.ee**2)
        nu = len(self.ii)-len(bp_vals)
        red_chisq = chisq/nu
        self.efac = np.sqrt(red_chisq)
        self.ee *= self.efac

    def get_xy(self):
        xy = [self.offs,self.offs]
        xy[(self.direction+1)%2] = np.zeros(len(self.offs))    # 0.0 is not accurate constant offset; gets fixed later!!
        return xy

    def clean_profs(self,profs):

        new_profs = profs

        # Fit order N polynomials to off-pulse bins of each subint.
        self.subtracted_polynomials = []
        for i,p in enumerate(profs):
            poly = np.poly1d(np.polyfit(self.phase[self.off],p[self.off],self.order))
            self.subtracted_polynomials.append(poly(self.phase))
            new_profs[i] = p-poly(self.phase)

        
        new_profs /= np.std(new_profs[:,self.off])
        self.prof_rms = np.array(self.subtracted_polynomials)

        return new_profs

    # radec_in = skycoord object, arc = quantity with units deg
    def directional_position_stuff(self,radec_in,arc):
        catalog_coord = radec_in
        
        # Direction = 1 (dec); no correction necessary.
        if self.direction:
            expected_start_ra = catalog_coord.ra
            expected_start_dec = Latitude(catalog_coord.dec)-Angle(0.5 * arc)
            recovered_skycoord_1d = self.get_recovered_skycoord(catalog_coord.ra)

        # Direction = 0 (ra); 
        else:
            cos_dec_factor = 1.0/np.cos(self.actual_start_coord.dec)
            expected_start_ra = Longitude(catalog_coord.ra)-Angle(0.5 * arc * cos_dec_factor)  # "arc" has units, so should be fine.
            expected_start_dec = catalog_coord.dec
            recovered_skycoord_1d = self.get_recovered_skycoord(catalog_coord.dec)

        self.expected_start_coord = SkyCoord(expected_start_ra,expected_start_dec,frame=ICRS,unit=(u.hourangle,u.deg),obstime="J2015.0")
       
        # asp = actual start position; isp = intended start position 
        self.asp_strs = (coord_string(self.actual_start_coord.ra,c_type='hms'), coord_string(self.actual_start_coord.dec,c_type='dms'))
        self.isp_strs = (coord_string(self.expected_start_coord.ra,c_type='hms'), coord_string(self.expected_start_coord.dec,c_type='dms'))
        self.systematic_offset = (self.expected_start_coord.separation(self.actual_start_coord)).to(u.arcmin).value
        self.systematic_pa = (self.expected_start_coord.position_angle(self.actual_start_coord)).to(u.deg).value

    def amps_errs(self,profs):
        # Based on simulations, amplitude errors should be proportional to std(off)/sqrt(N_on)
        self.ii = np.zeros(self.nsub) 
        self.ee = np.std(profs[:,self.off],axis=1)/np.sqrt(len(self.on))

        # Get amplitudes by fitting profile template to individual subints.
        for i,p in enumerate(profs):
            a_guess = max(p)-min(p)
            popt, pcov = curve_fit(scale_template, self.phase, p, p0=[a_guess])
            self.ii[i] = popt[0] 

        # Mask bad measurements using interquartile range (iqr) with logerrs.
        logerrs = np.log10(self.ee)
        mederr = np.median(logerrs)
        iqrerr = iqr(logerrs)
        t1 = mederr+2.0*iqrerr
        t2 = mederr-2.0*iqrerr
        for j in range(self.nsub):
            if logerrs[j] < t2:
                self.ee[j] = 999.0



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
        self.input_ra = ""
        self.input_dec = ""

        for line in f:
            if line.strip()[0] == '#': continue
            
            # Eventually, make this more generic for N scans in either RA/DEC directions.
            a = line.split('=')
            if a[0].count('Name'):
                self.name = str(a[1]).strip()
            if a[0].count('Arc'):
                self.arc = float(a[1])
            if a[0].count('Order'):
                self.order = float(a[1])
            if a[0].count('Threshold'):
                self.thresh = float(a[1])
            if a[0].count('Scan'):
                self.scan_files.append(self.path+'/'+str(a[1]).strip())
            if a[0].count('Template'):
                self.temp_files.append(self.path+'/'+str(a[1]).strip())
            # 0 = RA, 1 = Dec
            if a[0].count('Direction'):
                self.directions.append(int(a[1]))
            if a[0].count('Drift Rate'):
                self.driftrate = float(a[1])
            if a[0].count('Right Ascension'):
                self.input_ra = str(a[1]).strip()
            if a[0].count('Declination'):
                self.input_dec = str(a[1]).strip()

            else:
                continue

        f.close()

        self.actual_skycoord = SkyCoord(self.input_ra,self.input_dec, frame=ICRS, unit=(u.hourangle,u.deg), obstime="J2015.0")

        self.xx = np.empty(0)
        self.yy = np.empty(0)
        self.ii = np.empty(0)
        self.ee = np.empty(0)

        self.scans = []
        best_position = ['aa','dd']
        self.pos_errors = [999.0,999.0]

        # Now it's time to get relevant offset/intensity info.
        for ss,tt,dd in zip(self.scan_files, self.temp_files, self.directions):
            s_obj = scan(ss,tt,dd,self.driftrate,self.thresh,self.order)
            self.ii = np.append(self.ii,s_obj.ii)
            self.ee = np.append(self.ee,s_obj.ee)
            best_position[dd] = s_obj.best
            self.pos_errors[dd] = s_obj.bp_errs[1]    # Uncertainties in BP fit (0 = amplitude, 1 = offset)
            s_obj.directional_position_stuff(self.actual_skycoord,self.arc*u.deg)
            #self.scan_results(s_obj)
            self.scans.append(s_obj)

        self.x_start_position = SkyCoord(self.scans[0].asp_strs[0],self.scans[0].asp_strs[1],frame=ICRS,unit=(u.hourangle,u.deg),obstime="J2015.0")
        self.y_start_position = SkyCoord(self.scans[1].asp_strs[0],self.scans[1].asp_strs[1],frame=ICRS,unit=(u.hourangle,u.deg),obstime="J2015.0")
        self.center_position = SkyCoord(self.scans[1].asp_strs[0],self.scans[0].asp_strs[1],frame=ICRS,unit=(u.hourangle,u.deg),obstime="J2015.0")
        constant_xscan_offset = self.y_start_position.separation(self.center_position).arcmin
        constant_yscan_offset = self.x_start_position.separation(self.center_position).arcmin
        self.constant_scan_offsets = [constant_xscan_offset,constant_yscan_offset]

        for ss in self.scans:

            if not ss.direction:
                self.xx = np.append(self.xx,ss.xx)
                self.yy = np.append(self.yy,ss.yy+constant_yscan_offset)
            elif ss.direction:
                self.xx = np.append(self.xx,ss.xx+constant_xscan_offset)
                self.yy = np.append(self.yy,ss.yy)

            self.scan_results(ss)

        self.recovered_skycoord = SkyCoord(best_position[0],best_position[1],frame=ICRS,unit=(u.hourangle,u.deg),obstime="J2015.0")

        self.fit_2d_beam()
        self.text_out()

    def scan_results(self,scan_object):

        so = scan_object
        temp = so.temp_file.split('/')[-1].split('.')
        out_base = "%s_%s_results" % (temp[0],temp[1])
        #print out_base

        # Shift profile and fix on/off bins (won't work yet, need to fix external on/off in amps_errs function)
        bin_shift = so.nbin/2 - np.argmax(so.folded_profile)
        cleaned = np.roll(so.cleaned_profs,bin_shift,axis=1)
        folded = np.roll(so.folded_profile,bin_shift)
        on_bins = (so.on+bin_shift)%so.nbin
        off_bins = (so.off+bin_shift)%so.nbin

        fontsize = 10
        plt.rc('font',**{'family':'serif','serif':['Computer Modern']})
        plt.rc('xtick',labelsize=fontsize)
        plt.rc('ytick',labelsize=fontsize)
        plt.rc('text',usetex=True)

        fig = plt.figure(figsize=(4,6))
        gs = plt.GridSpec(4,4,hspace=0.15,wspace=0.0)
        ll_ax = fig.add_subplot(gs[1:,0:3])
        ul_ax = fig.add_subplot(gs[0,0:3])
        lr_ax = fig.add_subplot(gs[1:,3:4])

        # LL
        dir_string = so.get_dir_string()
        xylims = (0.0,1.0,0.0,so.length) #so.rate*so.length)
        ll_ax.imshow(cleaned,cmap=plt.cm.gray_r,origin='lower',interpolation='None',aspect='auto',extent=xylims)
        ll_ax.set_xlabel('Pulse Phase')
        ll_ax.set_ylabel('Integration Time (s)')
        
        # UL
        temp_prof = folded-min(folded)
        temp_prof /= max(temp_prof)
        ul_ax.plot(so.phase,temp_prof,c='black',lw=2)
        ul_ax.axes.get_xaxis().set_visible(False)
        ul_ax.axes.get_yaxis().set_visible(False)
        ul_ax.patch.set_visible(False)
        ul_ax.axis('off')
        #ul_ax.set_title('PSR %s (%s)' % (self.name,dir_string))
        

        # LR
        gi = (so.ee<500.0)    # Good indices
        bi = (so.ee>500.0)    # Bad indices
        lr_ax.errorbar(so.ii[gi],so.offs[gi],xerr=so.ee[gi],fmt='.',capsize=3,c='black',label='Measured Amplitudes')
        lr_ax.scatter(so.ii[bi],so.offs[bi],marker='x',c='gray',label='Rejected Measurements')
        lr_ax.plot(gaussian_beam(so.offs,so.bp_vals[0],so.bp_vals[1]),so.offs,ls='--',c='gray',label='Beam Model')
        #lr_ax.axes.get_yaxis().set_visible(False)
        lr_ax.xaxis.tick_top()
        lr_ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        lr_ax.yaxis.tick_right()
        lr_ax.set_ylim([0.0,so.rate*so.length])    # Lines up y-axes for LL/LR plots given "xylims"
        lr_ax.set_xlabel('Intensity')
        lr_ax.xaxis.set_label_position('top')
        lr_ax.set_ylabel('Offset from start %s (arcmin)' % (dir_string),rotation=270.0)
        lr_ax.yaxis.set_label_position('right')

        #lr_ax.legend(loc=1)
        #lr_ax.xaxis.set_label_position('top')
        #lr_ax.set_xlabel('Intensity (arbitrary units)')

        plt.savefig(out_base+'.pdf',format='pdf',bbox_inches='tight',dpi=300)


    def text_out(self):

        for ii,ss in enumerate(self.scans):
            print ""
            print "== %s scan (#%s) ==" % (ss.get_dir_string(),ii+1) 
            print "Angular coverage (deg): %s" % (self.arc)
            print "Mapping rate ('/s): %.3f" % (self.driftrate)
            print "Intended scan length (s): %.1f" % (self.arc*60.0/self.driftrate)
            print "Actual scan length (s): %.1f" % (ss.length)
            print "Effective angular difference ('): %.1f" % (self.arc*60.0 - ss.length*self.driftrate)
            dx_sys,dy_sys = ss.systematic_offset * np.sin(ss.systematic_pa * np.pi/180.0), ss.systematic_offset * np.cos(ss.systematic_pa * np.pi/180.0)
            print "Systematic RA offset ('): %.2f" % (dx_sys)
            print "Systematic Dec offset ('): %.2f" % (dy_sys)
            print "--------------------------------------"
            print "Intended Start Position: %s %s" % ss.isp_strs 
            print "Actual Start Position: %s %s" % ss.asp_strs 
            print "Amplitude Error EFAC: %.2f" % (ss.efac)
            print "1D Recovered %s: %s" % (ss.get_dir_string(),ss.best)
            print ""
            print ""
            gi = ss.ee < 500.0
            AA = np.max(ss.ii[gi])
            MM = np.sqrt(np.mean(ss.ee[gi]**2))
            hh = 60.0*self.arc/ss.nsub
            print "A:                        %.2f" % (AA)
            print "mu:                       %.2f" % (MM)
            print "Expected Uncertainty ('): %.2f" % (np.sqrt(2/np.pi)*hh*MM/AA)

        print ""
        print "Source Position:    %s %s" % (self.input_ra, self.input_dec)
        print "OTF Map Center:     %s %s" % (coord_string(self.center_position.ra,c_type='hms'),
                                            coord_string(self.center_position.dec,c_type='dms'))
        print "Recovered Position: %s %s" % (coord_string(self.recovered_skycoord.ra,c_type='hms'),
                                            coord_string(self.recovered_skycoord.dec,c_type='dms'))

        ra_err_1d  = self.pos_errors[0]
        dec_err_1d = self.pos_errors[1]
        sep_1d     = (self.actual_skycoord.separation(self.recovered_skycoord)).to(u.arcmin).value
        posang_1d  = (self.actual_skycoord.position_angle(self.recovered_skycoord)).to(u.deg).value

        print "RA/Dec Uncertainties ('): %.2f/%.2f" % (ra_err_1d,dec_err_1d)
        print "Separation ('):           %.2f" % (sep_1d)
        print "Position Angle (deg):     %.2f" % (posang_1d)

        print ""
        print "== 2D Fit Results =="
        # ress[0] = amplitude, ress[1] = ra offset, ress[2] = dec offset
        cos_dec_factor = 1.0/np.cos(self.center_position.dec)
        recovered_x_2d = self.x_start_position.ra+Angle(self.ress[1] * cos_dec_factor * u.arcmin)
        recovered_y_2d = self.y_start_position.dec+Angle(self.ress[2] * u.arcmin)
        recovered_skycoord_2d = SkyCoord(recovered_x_2d,recovered_y_2d,frame=ICRS,unit=(u.hourangle,u.deg),obstime="J2015.0")
        ra_err_2d, dec_err_2d = self.errs[1],self.errs[2]
        sep_2d = self.actual_skycoord.separation(recovered_skycoord_2d).arcmin
        posang_2d = self.actual_skycoord.position_angle(recovered_skycoord_2d).deg
        print "Recovered Position:       %s %s" % (coord_string(recovered_skycoord_2d.ra,c_type='hms'),
                                                coord_string(recovered_skycoord_2d.dec,c_type='dms')) 
        print "RA/Dec Uncertainties ('): %.2f/%.2f" % (ra_err_2d,dec_err_2d)
        print "Separation ('):           %.2f" % (sep_2d)
        print "Position Angle (deg):     %.2f" % (posang_2d)

        print ""
        print '1D:',sep_1d, posang_1d, ra_err_1d, dec_err_1d
        print '2D:',sep_2d, posang_2d, ra_err_2d, dec_err_2d


    def fit_2d_beam(self):
        amplitude_guess = np.max(self.ii)

        # Bootstrap...
        a_vals = []    # Amplitude values
        r_vals = []    # RA
        d_vals = []    # Dec
        ntrials = 10000
        nmeas = len(self.ii)

        for k in range(ntrials):
            bi = np.random.randint(0,high=nmeas,size=nmeas)    # Bootstrap indices
            bp2d_popt, bp2d_pcov = curve_fit(gaussian_beam_2d,(self.xx[bi],self.yy[bi]),self.ii[bi],p0=[amplitude_guess,60.0,60.0],sigma=self.ee[bi],absolute_sigma=True)
            a_vals.append(bp2d_popt[0])
            r_vals.append(bp2d_popt[1])
            d_vals.append(bp2d_popt[2])

        self.ress = [np.mean(a_vals),np.mean(r_vals),np.mean(d_vals)]
        self.errs = [np.std(a_vals),np.std(r_vals),np.std(d_vals)]
        # END OF BOOTSTRAP


# Running from the command line...
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('loc_file')
    args = parser.parse_args()
    x = loc_info(args.loc_file) 
