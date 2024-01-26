#=  cartesian rank
    i(0)---------------
    |   0   1   2   3  |
    |   4   5   6   7  |
    |   8   9  10  11  |
    |  12  13  14  15  |
    ---------------- j(1)
=#

using MPI
using HDF5

const Nx::Int64 = 256
const Ny::Int64 = 256

const Nproc_x::Int64 = 4
const Nproc_y::Int64 = 4
const Nghost::Int64 = 1
const root::Int64 = 0
const Iperiodic = (false, false)

# calculate local indices w/o ghost cells
const lo = [1+Nghost,1+Nghost]
const hi = [Nx ÷ Nproc_x+Nghost, Ny ÷ Nproc_y+Nghost]
const lo_g = [1, 1]
const hi_g = [hi[1]+Nghost, hi[2]+Nghost]

function laplacian(ϕ, ϕn, lo, hi)
    for j ∈ lo[2]:hi[2], i ∈ lo[1]:hi[1]
        @inbounds ϕn[i,j] = 0.25*(ϕ[i-1,j]+ϕ[i+1,j]+ϕ[i,j-1]+ϕ[i,j+1])
    end
end

function fill_boundary(ϕ, rank, Nproc_x)
    if rank ÷ Nproc_x == 0
        ϕ[1, :] .= 10.0
    end
end

function exchange_ghost(ϕ, comm_cart, comm, lo, hi, lo_g, hi_g)
    # exchange boundary
    src, dst = MPI.Cart_shift(comm_cart, 0, 1)
    @show src
    sendbuf = @view ϕ[hi[1]-Nghost+1:hi[1], :]
    recvbuf = @view ϕ[lo_g[1]:lo[1]-1, :]
    status = MPI.Sendrecv!(sendbuf, recvbuf, comm; dest=dst, source=src)

    src, dst = MPI.Cart_shift(comm_cart, 0, -1)
    sendbuf = @view ϕ[lo[1]:lo[1]+Nghost-1, :]
    recvbuf = @view ϕ[hi[1]+1:hi_g[1], :]
    status = MPI.Sendrecv!(sendbuf, recvbuf, comm; dest=dst, source=src)

    src, dst = MPI.Cart_shift(comm_cart, 1, 1)
    sendbuf = @view ϕ[:, hi[2]-Nghost+1:hi[2]]
    recvbuf = @view ϕ[:, lo_g[2]:lo[2]-1]
    status = MPI.Sendrecv!(sendbuf, recvbuf, comm; dest=dst, source=src)

    src, dst = MPI.Cart_shift(comm_cart, 1, -1)
    sendbuf = @view ϕ[:, lo[2]:lo[2]+Nghost-1]
    recvbuf = @view ϕ[:, hi[2]+1:hi_g[2]]
    status = MPI.Sendrecv!(sendbuf, recvbuf, comm; dest=dst, source=src)
end

function run()
    MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    Nprocs = MPI.Comm_size(comm)
    nIter = 1

    if Nprocs != Nproc_x * Nproc_y
        error("Not correct processes")
    end
    comm_cart = MPI.Cart_create(comm, [Nproc_x, Nproc_y]; periodic=Iperiodic)

    # initialize
    ϕ = ones(Float64, hi_g[1], hi_g[2])
    # fill boundary
    fill_boundary(ϕ, rank, Nproc_x)
    ϕn = copy(ϕ)

    @time for _ ∈ 1:nIter
        laplacian(ϕ, ϕn, lo, hi)

        ϕ, ϕn = ϕn, ϕ

        exchange_ghost(ϕ, comm_cart, comm, lo, hi, lo_g, hi_g)

        MPI.Barrier(comm)

        fill_boundary(ϕ, rank, Nproc_x)
    end

    ϕng = @view ϕ[lo[1]:hi[1], lo[2]:hi[2]] # remove ghost
    h5open("test.h5", "w", comm) do f
        dset = create_dataset(
            f,
            "/phi",
            datatype(Float64),
            dataspace(hi[1]-Nghost, hi[2]-Nghost, Nprocs);
            chunk=(hi[1]-Nghost, hi[2]-Nghost, 1),
            dxpl_mpio=:collective
        )
        dset[:, :, rank + 1] = ϕng
    end

    MPI.Finalize()
end

run()
