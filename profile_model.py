import numpy as np
import matplotlib.pyplot as plt
from read_gaussfit import *

#def gaussian(peak_phase,phase_array,fwhm,amp):
#  return amp * np.exp(-2.7726 * (peak_phase - phase_array)**2 / (fwhm * fwhm))

#phase = np.arange(1000)/1000.
#g = gaussian(0.78288,phase,0.02560,1388526.8)

nbins = 64

template = read_gaussfit('1921.gaussians',nbins)

print np.where(template > 0.0001*np.max(template))

#plt.plot(template)
#plt.show()
