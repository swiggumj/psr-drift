from test_pfd import *
import sys

if '-h' in sys.argv:
  print 'Usage: python pfd_offset.py -d [driftrate] -p [gaussians file] [pfd file]'
  sys.exit()

# Get drift rate.
if '-d' in sys.argv:
  d_ind = int(sys.argv.index('-d'))+1
  driftrate = float(sys.argv[d_ind])
else:
  print 'Indicate drift rate (arcmin/sec) with -d!'
  exit()

# Get profile model.
if '-p' in sys.argv:
  p_ind = int(sys.argv.index('-p'))+1
  profile = str(sys.argv[p_ind])
else:
  print 'Indicate profile model filename with -p!'
  exit()

# Check that last argument is the .pfd file.
if '.pfd' in sys.argv[-1]:
  pfd_file = str(sys.argv[-1])
  x = get_bpo(pfd_file,profile,dr=driftrate) 
  x.fit_bp()
  x.plot_fit()
else:
  print 'The last argument should be a .pfd file.'
  exit()
