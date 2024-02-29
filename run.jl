# Only 1D-x direction MPI for now
include("solver.jl")
import Adapt

# LES
const LES_smag::Bool = false       # if use Smagorinsky model
const LES_wale::Bool = false        # if use WALE model

# flow control
const Nprocs::Int64 = 1              # number of GPUs
const dt::Float64 = 2e-8             # dt for simulation, make CFL < 1
const Time::Float64 = 1e-3           # total simulation time
const maxStep::Int64 = 100000         # max steps to run

const plt_out::Bool = true           # if output plt file
const step_plt::Int64 = 100          # how many steps to save plt
const plt_compress_level::Int64 = 1  # output file compression level 0-9, 0 for no compression

const chk_out::Bool = true           # if checkpoint is made on save
const step_chk::Int64 = 500          # how many steps to save chk
const chk_compress_level::Int64 = 1  # checkpoint file compression level 0-9, 0 for no compression
const restart::String = "none"     # restart use checkpoint, file name "*.h5" or "none"

# do not change 
const Ncons::Int64 = 5 # ρ ρu ρv ρw E 
const Nprim::Int64 = 7 # ρ u v w p T ei
# load mesh info
const NG::Int64 = h5read("metrics.h5", "NG")
const Nx::Int64 = h5read("metrics.h5", "Nx")
const Ny::Int64 = h5read("metrics.h5", "Ny")
const Nz::Int64 = h5read("metrics.h5", "Nz")
const Nxp::Int64 = Nx ÷ Nprocs # make sure it is integer
# here we use 256 threads/block and limit registers to 255
const maxreg::Int64 = 255
const nthreads::Tuple{Int32, Int32, Int32} = (4, 8, 8)
const nblock::Tuple{Int32, Int32, Int32} = (cld((Nxp+2*NG), 4), 
                                            cld((Ny+2*NG), 8),
                                            cld((Nz+2*NG), 8))


struct constants{T, VT}
    Rg::T
    gamma::T
    C_s::T
    T_s::T
    Pr::T
    Cp::T
    CD4::VT
    Hybrid::VT
    WENO5::VT # eps, 13/12, 1/6
    TENO5::VT # eps, CT, 1/6
    UP7::VT
end

Adapt.@adapt_structure constants

# Run the simulation
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nGPU = MPI.Comm_size(comm)
if nGPU != Nprocs && rank == 0
    error("Oops, nGPU ≠ $Nprocs\n")
end
# set device on each MPI rank
device!(rank)
# constant parameters
const consts = constants(287.0, 1.4, 1.458e-6, 110.4, 0.72, 1004.5, 
         CuArray([2/3, 1/12]),
         CuArray([1e-4, 0.2]),
         CuArray([1e-40, 13/12, 1/6]),
         CuArray([1e-40, 1e-5, 1/6]),
         CuArray([-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420]))

CUDA.@time time_step(rank, comm, consts)

MPI.Finalize()
