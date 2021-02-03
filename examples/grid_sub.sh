#!/bin/bash
#
#SBATCH --job-name=example
#SBATCH --output=output.txt
#SBATCH --partition=2hr

python3 example_grid.py $1 $2 $3 $4
