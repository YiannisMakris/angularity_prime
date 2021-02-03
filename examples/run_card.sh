#!/bin/bash

# run this file for parallel fits with random profiles. 
# In terminal:  bash run_card.sh

#----------------   USER INPUT PARAMETERS ----------------
loc=0			  	# which angularity (=0 -> a=-1,  =1 -> a=-0.75, ... )
min=4			  	# starting bin for the analysis 
num=9 				# bumber of bins to be included in the analysis 
NumberOfFits=10	  	# Number of fits to be performed based on random profile choises  
# dir="../outputs/min"$min"_num"$num"_a"$loc"/"   # Directory for saving the output result
dir=../outputs/example_cluster/   # Directory for saving the output result
#---------------------------------------------------------

mkdir $dir

for ((run = 0 ; run < $NumberOfFits; run++))
do
	sbatch fits_sub.sh $loc $min $num $run $dir
done