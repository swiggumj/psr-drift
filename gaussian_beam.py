from numpy import exp

# Model of beam shape (gaussian)
def gaussian_beam(rr,amp,offset,fwhm=36.0,obsfreq=350.0):
  return amp *  exp(-2.7726 * (rr - offset)**2 / (fwhm * fwhm))
