include("solver.jl")

# LES
const LES_smag::Bool = false       # if use Smagorinsky model
const LES_wale::Bool = false        # if use WALE model

# thermal state
const γ::Float32 = 1.4
const Rg::Float32 = 287
const Cp::Float32 = Rg*γ/(γ-1)
const C_s::Float32 = 1.458f-6
const T_s::Float32 = 110.4
const Pr::Float32 = 0.72

# flow control
const mesh::String = "mesh.h5"
const metrics::String = "metrics.h5"
const Nprocs::SVector{3, Int64} = [1,1,1] # number of GPUs
const Iperiodic = (false, false, false)   # periodic direction

const LTS::Bool = false              # if use LTS (local time stepping)
const LTS_CFL::Float32 = 0.5f0       # LTS auto Δt CFL
const dt::Float32 = 3f-8             # dt for simulation, make CFL < 1
const Time::Float32 = 1f-3           # total simulation time
const maxStep::Int64 = 1000         # max steps to run

const plt_xdmf::Bool = true         # if use HDF5+XDMF for plt output
const plt_out::Bool = true           # if output plt file
const step_plt::Int64 = 100          # how many steps to save plt
const plt_shuffle::Bool = true       # shuffle to make compress more efficient
const plt_compress_level::Int64 = 1  # output file compression level 0-9, 0 for no compression

const chk_out::Bool = false           # if checkpoint is made on save
const step_chk::Int64 = 2000          # how many steps to save chk
const chk_shuffle::Bool = true       # shuffle to make compress more efficient
const chk_compress_level::Int64 = 1  # checkpoint file compression level 0-9, 0 for no compression
const restart::String = "none"     # restart use checkpoint, file name "*.h5" or "none"

const average::Bool = false                 # if do average
const avg_step::Int64 = 10                  # average interval
const avg_total::Int64 = 1000               # total number of samples
const avg_shuffle::Bool = true       # shuffle to make compress more efficient
const avg_compress_level::Int64 = 1  # output file compression level 0-9, 0 for no compression

const sample::Bool = false                             # if do sampling (slice)
const sample_step::Int64 = 10                          # sampling interval
const sample_index::SVector{3, Int64} = [-1, 130, -1]  # slice index in 3 directions, -1 for no slicing

# filtering
const filtering::Bool = false              # if do filtering
const filtering_nonlinear::Bool = false    # if filtering is shock capturing
const filtering_interval::Int64 = 100      # filtering step interval
const filtering_rth::Float32 = 1f-5        # filtering threshold for nonlinear
const filtering_s0::Float32 = 1.f0         # filtering strength

# do not change 
const Ncons::Int64 = 5 # ρ ρu ρv ρw E 
const Nprim::Int64 = 6 # ρ u v w p T
# scheme constant
const character::Bool = true        # Characteristic-wise reconstruction or not
const splitMethod::String = "SW"    # options are: SW, LF, VL, AUSM
const hybrid_ϕ1::Float32 = 5f-2     # < ϕ1: UP7
const hybrid_ϕ2::Float32 = 1.f0     # < ϕ2: WENO7 in FP64
const hybrid_ϕ3::Float32 = 10.f0    # < ϕ3: WENO5, else NND2
const Linear_ϕ::Float32 = 1.f0     # dissipation control for linear scheme
# adjust this to get mixed upwind-central linear scheme
const UP7::SVector{7, Float32} = SVector(-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420)
const CD6::SVector{7, Float32} = SVector(0, 1/60, -2/15, 37/60, 37/60, -2/15, 1/60)
const Linear::SVector{7, Float32} = UP7 * Linear_ϕ + CD6 * (1.f0 - Linear_ϕ)

# load mesh info
const NG::Int64 = h5read(mesh, "NG")
const Nx::Int64 = h5read(mesh, "Nx")
const Ny::Int64 = h5read(mesh, "Ny")
const Nz::Int64 = h5read(mesh, "Nz")
const Nxp::Int64 = Nx ÷ Nprocs[1] # make sure it is integer
const Nyp::Int64 = Ny ÷ Nprocs[2] # make sure it is integer
const Nzp::Int64 = Nz ÷ Nprocs[3] # make sure it is integer
# here we use 512 threads/block and limit registers to 128
const maxreg::Int64 = 128
const nthreads::Tuple{Int32, Int32, Int32} = (8, 8, 8)
const nblock::Tuple{Int32, Int32, Int32} = (cld((Nxp+2*NG), 8), 
                                            cld((Nyp+2*NG), 8),
                                            cld((Nzp+2*NG), 8))

# For simple kernel without register limit
const nthreads2::Tuple{Int32, Int32, Int32} = (16, 8, 8)
const nblock2::Tuple{Int32, Int32, Int32} = (cld((Nxp+2*NG), 16), 
                                             cld((Nyp+2*NG), 8),
                                             cld((Nzp+2*NG), 8))
# Run the simulation
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nGPU = MPI.Comm_size(comm)
shmcomm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
local_rank = MPI.Comm_rank(shmcomm)
comm_cart = MPI.Cart_create(comm, Nprocs; periodic=Iperiodic)
if nGPU != prod(Nprocs) && rank == 0
    error("Oops, nGPU ≠ $Nprocs\n")
end
# set device on each MPI rank
# device!(local_rank)

time_step(rank, comm_cart)

MPI.Finalize()
