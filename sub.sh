#!/bin/bash
#SBATCH -J Flame3D
#SBATCH -n 16
#SBATCH -N 4
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-socket=1
#SBATCH -p blcy
#SBATCH --mem=64G
#SBATCH -o slurm-out
#SBATCH -e slurm-err

module purge; 
module load dtk/23.04.1 compiler/cmake/3.24.1 compiler/devtoolset/7.3.1 mpi/hpcx/2.7.4/gcc-7.3.1

mpirun -np 16 julia run.jl
