using MPI
using HDF5

const NG = h5read("../metrics.h5", "NG")
const Nx = h5read("../metrics.h5", "Nx")
const Ny = h5read("../metrics.h5", "Ny")
const Nz = h5read("../metrics.h5", "Nz")
const Nprocs = 4

function test()
    MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    Nprocs = MPI.Comm_size(comm)
    fid = h5open("../metrics.h5", "r")
    Jp = fid["J"]
    Jpp = Jp[rank+1, rank+1, rank+1]
    close(fid)
    @show Jpp
end

@time test()