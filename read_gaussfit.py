import numpy as np
from psr_utils import gaussian_profile
# Slightly modified from version in psr_utils (PRESTO)
# In this case, I want to preserve the model's phase.

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
    template = np.zeros(proflen, dtype='d')
    for ii in range(len(ampls)):
        template += ampls[ii]*gaussian_profile(proflen, phass[ii], fwhms[ii])
    return template

