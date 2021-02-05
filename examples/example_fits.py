#------------------------------------------------------------------------------
#  Run this example on a cluster for multiple fits using random profiles. 
#  For SLURM workload manager we also provide an example run_card.sh for 
#  submiting multiple jobs with specific user-parameters. We provide 
#  a python script collect.py for collecting the results from the different 
#  run in a single file. 
#------------------------------------------------------------------------------

import sys
sys.path.append('../python/')
from main import *
from param import *
import os
import time
import numpy as np
import random as rnd
from iminuit import Minuit

# Will overwrite the values passed from param.py
# if (sys.argv[1].isnumeric()): loc         = int(sys.argv[1])
# if (sys.argv[2].isnumeric()): bin_min     = int(sys.argv[2]) 
# if (sys.argv[3].isnumeric()): num_of_bins = int(sys.argv[3])

# Other variable assosiated with the run 
rnd_seed    = int( sys.argv[1])
directory   = 	   sys.argv[2]
job_id      = os.environ.get('SLURM_JOB_ID')

# First run is assgned the default choice of profiles i.e., "rnd_seed = -1"

bins, Y_exp, V_inv = experimental_input_all_a(bin_min_all_a, num_of_bins_all_a)

chi = set_chi_all_a(Q, bins, Y_exp, V_inv, rnd_seed)

print("Minuit: " )

m = Minuit(chi, 0.11, 0.4)
m.migrad()  # run optimiser 

fit = np.append(np.array(m.values), rnd_seed )

out_name = directory + "fits_"+ job_id +".txt"

np.savetxt(out_name,  fit , delimiter='\t')   # export data    

