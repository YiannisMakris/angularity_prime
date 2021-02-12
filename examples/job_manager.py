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
        if       (vec[0] == "loc" ): loc = vec[1] 
        elif (vec[0] == "min" ): bin_min = vec[1]
        elif (vec[0] == "num" ): num_of_bins = vec[1]
        elif (vec[0] == "run" ): runID = vec[1]
        elif (vec[0] == "grid"): gridSize = int(vec[1])
        else: print ("Option '" + vec[0] + "' is not a valid option!")

directory = "../outputs/singleA_default_C_Rmax100/"

mkdirCmd = ["mkdir",  directory]
subprocess.run(mkdirCmd)


order = 0
for a_Z in linspace(0.1, 0.12, num = gridSize, endpoint = True):
        for Omega1 in linspace(0.11, 1.0, num = gridSize, endpoint = True):

                order += 1

                bashCmd = ["sbatch", "grid_sub.sh", str(a_Z), str(Omega1), str(order), directory]
                # run bash command
                subprocess.run(bashCmd)

                time.sleep(0.01)