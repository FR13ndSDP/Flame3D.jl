# Flame3D.jl
> **FL**uid **A**nd **M**achine learning **E**ngine **3D**

Code for compressible flow simulation with neural network for real gas and chemical reaction.

- `CUDA.jl` and `MPI.jl` for multi-GPU parallelization
- 3D, with high order scheme (up to 7th order)
- `Lux.jl` trained model combined with `Cantera`

## Usage

Generate mesh and metrics with
```
julia parse_mesh.jl
```
It will compute metrics and store it to `metrics.h5`, if you want to visualize the mesh, make sure `const vis::Bool = true`, then the `mesh.vts` file can be opened with Paraview, etc.

Run the simulation with 
```
julia run.jl
```
Or to use multi-GPU, make `Nprocs=<nprocs>` and run it with `mpiexecjl -n <nprocs> julia run.jl`.


For more details, checkout the comments in `run.jl`.

## Validation cases

### Taylor Green Vortex
![TGV](./Utils/TGV.png)

### Combustion
![JET](./Utils/JET.png)