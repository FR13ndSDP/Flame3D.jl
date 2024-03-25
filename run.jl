# Only 1D-x direction MPI for now
include("solver.jl")
using PyCall
import Adapt

# LES
const LES_smag::Bool = false       # if use Smagorinsky model
const LES_wale::Bool = false       # if use WALE model

# reaction
const reaction::Bool = false       # if reaction is activated
const T_criteria::Float64 = 500.0  # reaction temperature criteria 
const Nspecs::Int64 = 5            # number of species
const Nreacs::Int64 = 5           # number of reactions, consistent with mech
const mech::String = "./NN/Air/air.yaml" # reaction mechanism file in cantera format

const Luxmodel::Bool = false       # if use Neural network model

const Cantera::Bool = false         # if use Cantera
const nthreads_cantera::Int64 = 24  # Cantera openmp threads

const stiff::Bool = true           # if reaction is stiff
const sub_step::Int64 = 1          # reaction substep in stiff case

# IBM
const IBM::Bool = true

# flow control
const Nprocs::Int64 = 1              # number of GPUs
const dt::Float64 = 1e-6             # dt for simulation, make CFL < 1
const Time::Float64 = 0.1           # total simulation time
const maxStep::Int64 = 100         # max steps to run

const plt_out::Bool = true           # if output plt file
const step_plt::Int64 = 200          # how many steps to save plt
const plt_compress_level::Int64 = 1  # output file compression level 0-9, 0 for no compression

const chk_out::Bool = true           # if checkpoint is made on save
const step_chk::Int64 = 100          # how many steps to save chk
const chk_compress_level::Int64 = 1  # checkpoint file compression level 0-9, 0 for no compression
const restart::String = "none"     # restart use checkpoint, file name "*.h5" or "none"

const average = false                 # if do average
const avg_step = 10                  # average interval
const avg_total = 100                # total number of samples

# do not change unless you know what you are doing
const Ncons::Int64 = 5 # ρ ρu ρv ρw E 
const Nprim::Int64 = 7 # ρ u v w p T ei
# scheme constants
const SWϵ2::Float64 = 0
const WENOϵ::Float64 = 1e-26
const TENOϵ::Float64 = 1e-40
const TENOct::Float64 = 1e-5
const hybrid_ϕ1::Float64 = 1e-3
const hybrid_ϕ2::Float64 = 0.1
const UP7::SVector{7, Float64} = SVector(0.0, 17/600, -23/120, 22/30, 0.5, -0.075, 0.005)
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

struct thermoProperty{RT, VT, MT, TT}
    Ru::RT
    min_temp::RT
    max_temp::RT
    mw::VT
    coeffs_sep::VT
    coeffs_lo::MT
    coeffs_hi::MT
    visc_poly::MT
    conduct_poly::MT
    binarydiff_poly::TT
end

struct reactionProperty{RT, IT, VT, MT}
    atm::RT
    reaction_type::IT
    sgm::IT
    vf::VT
    vr::VT
    Arr::MT
    ef::MT
    loP::MT
    Troe::MT
end

Adapt.@adapt_structure thermoProperty
Adapt.@adapt_structure reactionProperty

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
const thermo = initThermo(mech) # now only NASA7
const react = initReact(mech)

CUDA.@time time_step(rank, comm, thermo, react)

MPI.Finalize()
