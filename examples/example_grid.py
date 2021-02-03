#------------------------------------------------------------------------------
#  Run this example on a cluster for constructing a grid if the chi-squared 
#  using center profiles. 
#  For SLURM workload manager we also provide a python script "job_manager.py" 
#  and an example grid_sub.sh.
#------------------------------------------------------------------------------

import sys
sys.path.append('../python_single_a/')
from main import *
from param import *
import os
import time
import numpy as np
import random as rnd

# points on the grid 
a_Z 		= float( sys.argv[1] )
Omega1 		= float( sys.argv[2] )
# Other variable assosiated with the run 
run_id 		= int( sys.argv[3] )
directory   = 	   sys.argv[4]
job_id      = os.environ.get('SLURM_JOB_ID')

bins, Y_exp, V_inv = experimental_input_all_a( bin_min_all_a, num_of_bins_all_a)

rnd_seed = -1

chi = set_chi_all_a(Q, bins, Y_exp, V_inv, rnd_seed)

cst = chi(a_Z, Omega1)

grid_point = np.array([a_Z, Omega1, cst, run_id])

out_name = directory + "output_"+ job_id +".txt"

np.savetxt(out_name,  grid_point , delimiter='\t')   # export data    



