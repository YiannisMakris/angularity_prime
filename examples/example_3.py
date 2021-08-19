
import sys
import os
import numpy as np
import subprocess
import time
sys.path.append('../python/')
from main import *
from param import *
from numpy import linspace

# --------------------------------------------
gridSize = 51    # For LANL cluster use maximum = 31 (use = 1 for testing  )
rnd_seed = -1   
directory = "../outputs/example_seed_"+str(rnd_seed)+"/"
# --------------------------------------------

mkdirCmd = ["mkdir",  directory]
subprocess.run(mkdirCmd)

bins, Y_exp, V_inv = experimental_input_single_a(loc, bin_min, num_of_bins)

chi = set_chi_single_a(loc, Q, bins, Y_exp, V_inv, rnd_seed, mode = "shift")

all_outputs = []

run_id = 0
for a_Z in linspace(0.10, 0.12, num = gridSize, endpoint = True):
	for Omega1 in linspace(0.2, 1.0, num = gridSize, endpoint = True):
		run_id += 1
		cst = chi(a_Z, Omega1)
		grid_point = np.array([a_Z, Omega1, cst, run_id])
		all_outputs.append(grid_point)

all_outputs = np.array(all_outputs)

sorted_grid = all_outputs[np.argsort(all_outputs[:, 3])]

np.savetxt(directory + "grid_example.txt",  sorted_grid , delimiter='\t')   # export data  
