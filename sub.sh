#!/bin/bash
#SBATCH -J Flame3D
#SBATCH -n 16
#SBATCH -N 4
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-socket=1
#SBATCH -p bingxing-dcu
#SBATCH -o slurm-out
#SBATCH -e slurm-err

# remember to execute "conda activate ct-env" and "loadenv" before sbatch sub.sh
export LD_LIBRARY_PATH=;
module purge; 
module load compiler/rocm/dtk-23.10.1 compiler/cmake/3.28.0 compiler/devtoolset/7.3.1 mpi/hpcx/2.7.4/gcc-7.3.1


mpirun -np 16 julia run.jl