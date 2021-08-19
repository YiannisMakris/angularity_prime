# ------------------------------------------------------------------------------
#  This is an example code of how to use the angularities prime package along 
#  with imunuit in order to perform a fit of the strong coupling constant an 
#  the hadronization parameter Omega_1. The output of the fit is written in the 
#  relative path: ../outputs/example/example_output.txt. 
# ------------------------------------------------------------------------------

import sys
sys.path.append('../python/')
from main import *
from param import *
from newton_optimize import *
import subprocess
import numpy as np

rnd_seed = -1  # Uses the central profile. For a random profile change this value to a possitive integer. 

# -----   For fitting a single value of a -----
bins, Y_exp, V_inv = experimental_input_single_a(loc, bin_min, num_of_bins)

chi = set_chi_single_a(loc, Q, bins, Y_exp, V_inv, rnd_seed, mode = "shift")

print( "This will take several minutes")

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++" )

print("Minuit: " )

m = Minuit(chi, 0.11, 0.4)
m.migrad()  # run optimiser 

fit = np.append(np.array(m.values), rnd_seed )

print ("a_s(m_Z) = ", round(fit[0], 5) )
print ("Omega_1  = ", round(fit[1], 5) )

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++" )

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++" )

print("In house minimizer: " )

fit = np.append(np.array(newton(chi, 0.11, 0.4)[0]), rnd_seed )

print ("a_s(m_Z) = ", round(fit[0], 5) )
print ("Omega_1  = ", round(fit[1], 5) )

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++" ) 