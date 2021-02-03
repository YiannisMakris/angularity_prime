import glob
import numpy as np

directory = "../outputs/example_cluster/"

input_name = directory + "output_*.txt"

list=glob.glob(input_name)

all_outputs = []

for file in list:
	 
	data = loadtxt(file, comments="#", delimiter="\t", unpack=False)

	all_outputs.append(data)

	# os.remove(file)

all_outputs = np.array(all_outputs)

np.savetxt(directory +  "collect.txt",  all_outputs , delimiter='\t')   # export data  