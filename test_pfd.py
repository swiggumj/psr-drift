import prepfold
import numpy as np
import matplotlib.pyplot as plt


def test(pfd_fname,nsubints=None,dr=0.666):

    pf = prepfold.pfd(pfd_fname)
    pf.dedisperse()

    # BEST WAY TO DO THIS?? 
    nsub    = pf.npart
    nchan   = 1
    nbin    = pf.proflen
  
    if nsubints and nsub>nsubints:

        if nsub%nsubints:
            newsubints = float(nsub)/(int(nsub)/int(nsubints))

        else: newsubints = nsubints
    
        profs = pf.combine_profs(newsubints,nchan)

    elif nsub<nsubints:
        print "Cannot scrunch to %d subints since original file has %d subints." % (nsubints,nsub)

    else: newsubints = nsub
  
    profs = pf.combine_profs(newsubints,nchan)

    mid_time = (pf.mid_secs[0]+pf.mid_secs[-1])/2.0
    times = np.linspace(pf.mid_secs[0],pf.mid_secs[-1],newsubints) - mid_time
    offsets_arcmin = times*dr    # dr is in arcmin/second

    print ""
    print "Time (mid) first subint:    %s" % (pf.mid_secs[0])
    print "Time (mid) last subint:     %s" % (pf.mid_secs[-1])
    print "Time half-way through scan: %s" % (mid_time)
    print ""
    print "Times array (seconds):"
    print times
    print ""
    print "Offsets array (arcmin):"
    print offsets_arcmin
    print ""

if __name__ == "__main__":

    test('PSR_1921+2153_RA-dr_0.666.pfd',dr=0.666)
