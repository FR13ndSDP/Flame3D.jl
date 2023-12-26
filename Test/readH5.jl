#=  cartesian rank
    i(0)---------------
    |   0   1   2   3  |
    |   4   5   6   7  |
    |   8   9  10  11  |
    |  12  13  14  15  |
    ---------------- j(1)
=#

using HDF5, Plots

f = h5open("test.h5", "r")
obj = f["/phi"]
a = read(obj)

Nx = 256
Ny = 256
Nprocs = 4
Nproc_x = 2
Nproc_y = 2

ϕ = zeros(Float64, Nx, Ny)
ϕ[1:128, 1:128] = a[:, :, 1]
ϕ[1:128, 129:256] = a[:, :, 2]
ϕ[129:256, 1:128] = a[:, :, 3]
ϕ[129:256, 129:256] = a[:, :, 4]
heatmap(ϕ)