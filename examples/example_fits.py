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

# Will overwrite the values passed from param.py
if (sys.argv[1].isnumeric()): loc         = int(sys.argv[1])
if (sys.argv[2].isnumeric()): bin_min     = int(sys.argv[2]) 
if (sys.argv[3].isnumeric()): num_of_bins = int(sys.argv[3])

# Other variable assosiated with the run 
run_id 		= int( sys.argv[4] )
directory   = 	   sys.argv[5]
job_id      = os.environ.get('SLURM_JOB_ID')

# First run is assgned the default choice of profiles i.e., "rnd_seed = -1"
if (run_id == 0): rnd_seed = -1
else: rnd_seed = rnd.randint(1,1000000000)

bins, Y_exp, V_inv = experimental_input_single_a(loc, bin_min, num_of_bins)

chi = set_chi_single_a(loc, Q, bins, Y_exp, V_inv, rnd_seed)

print("Minuit: " )

m = Minuit(chi, 0.11, 0.4)
m.migrad()  # run optimiser 

fit = np.append(np.array(m.values), rnd_seed )

out_name = directory + "output_"+ job_id +".txt"

np.savetxt(out_name,  fit , delimiter='\t')   # export data    

