# Flame3D.jl
> **FL**uid **A**nd **M**achine learning **E**ngine **3D**

Code for compressible flow simulation with neural network for real gas and chemical reaction.

- `CUDA.jl` and `MPI.jl` for multi-GPU parallelization
- 3D, with high order scheme (up to 7th order)
- `Lux.jl` trained model combined with `Cantera`