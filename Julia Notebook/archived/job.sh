#!/bin/bash
#SBATCH -J lbfgs
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -o lbfgs.out

module load julia
julia lbfgs.jl
