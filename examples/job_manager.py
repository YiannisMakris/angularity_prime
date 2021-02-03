import subprocess
import sys
import glob
import os
import time
import numpy as np
import random as rnd
from numpy import loadtxt, linspace
from iminuit import Minuit
from scipy import interpolate

for arg in sys.argv[1:]:
	arg = arg.replace( "-", "")
	vec = arg.split("=")
	if 	 (vec[0] == "loc" ): loc = vec[1] 
	elif (vec[0] == "min" ): bin_min = vec[1]
	elif (vec[0] == "num" ): num_of_bins = vec[1]
	elif (vec[0] == "run" ): runID = vec[1]
	elif (vec[0] == "grid"): gridSize = int(vec[1])
	else: print ("Option '" + vec[0] + "' is not a valid option!")

directory 		= "../outputs/simfit_default_Rmax10/"

mkdirCmd = ["mkdir",  directory]
subprocess.run(mkdirCmd)


order = 0
for a_Z in linspace(0.1, 0.12, num = gridSize, endpoint = True):
	for Omega1 in linspace(0.11, 1.0, num = gridSize, endpoint = True):

		bashCmd = ["sbatch", "grid_sub.sh", str(a_Z), str(Omega1), str(order), directory]
		order += 1 
		subprocess.run(bashCmd)

		time.sleep(0.01)
		# run bash command


input_name = directory +  "output_*.txt"

list=glob.glob(input_name)

while (len(list) != gridSize**2 ):
	print ("Computattion is not completed. Next check in 5 secs.")
	print ("----------------------------------------------------")
	time.sleep( 5 )
	list=glob.glob(input_name)

print ("Computattion is now completed!")
time.sleep(1)

grid = [] 
for file in list:
	 
	data = loadtxt(file, comments="#", delimiter="\t", unpack=False)
	
	grid.append(data)

	os.remove(file)

grid = np.array(grid)
sorted_grid = grid[np.argsort(grid[:, 3])]
np.savetxt(eval_directory_main +  "grid.txt",  sorted_grid , delimiter='\t')   # export data  


sys.exit()