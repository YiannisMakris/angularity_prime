#!/bin/bash
#
#SBATCH --job-name=example
#SBATCH --output=output.txt
#SBATCH --partition=1day

python3 example_fits.py $1 $2 $3 $4 $5
