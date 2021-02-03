import numpy as np

#************  Fundamentals  *****************

NC = 3.0           	# Number of colors 

nf = 5.0           	# Number of quark flavors 

m_Z = 91.2         	# Z bosson mass

#**************  SU(N) Const. ****************

CF = (NC**2 - 1) / (2 * NC)

CA = NC

TF = 1.0 / 2

#************** Gap parameteres ****************

matched = True    	# match the cross-section to the full QCD in the tail

log_accuracy = 2   	# intiger values only! NNLL' := 2, (Currently is the only case supported)

initial_D = 0.1  	# = \bar{\Dellta} (R_\{Delta}, R_{\Delta}) in GeV

RD = 1.5         	# R_\{Delta} in GeV

loc = 0	            # Integer values only!  If "sim_fit = False" then specify which angularity 

avec = np.array(  [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5]  )	# List of values of a 

Q = m_Z	            # Center-of-mass energy

#************** Bin-selection ****************

bin_min_all_a      =  np.array(  [5, 5, 6, 7, 7, 8, 9]  )

num_of_bins_all_a  =  np.array(  [7, 7, 7, 8, 9, 9, 8]  )

bin_min            =  4

num_of_bins        =  9



