# Only 1D-x direction MPI for now
include("solver.jl")
using PyCall
import Adapt

# load mesh info
const NG = h5read("metrics.h5", "NG")
const Nx = h5read("metrics.h5", "Nx")
const Ny = h5read("metrics.h5", "Ny")
const Nz = h5read("metrics.h5", "Nz")

# global variables, do not change name
const reaction::Bool = false       # if reaction is activated
const Luxmodel::Bool = false       # if use Neural network model
const Cantera::Bool = false        # if use Cantera
const stiff::Bool = true           # if reaction is stiff
const sub_step::Int64 = 1          # reaction substep
const T_criteria::Float64 = 500.0  # reaction temperature criteria 
const dt::Float64 = 1e-8           # dt for simulation, make CFL < 1
const Time::Float64 = 1e-4         # total simulation time
const step_out::Int64 = 100        # how many steps to save result
const chk_out::Bool = true         # if checkpoint is made on save
const chk_compress_level = 3       # checkpoint file compression level 0-3, 0 for no compression
const restart::String = "none"     # restart use checkpoint, file name "chk**.h5"

const Nspecs::Int64 = 9 # number of species
const Ncons::Int64 = 5 # ρ ρu ρv ρw E 
const Nprim::Int64 = 7 # ρ u v w p T ei
const Nreacs::Int64 = 21 # number of reactions, consistent with mech
const mech::String = "./NN/H2/LiDryer.yaml" # reaction mechanism file in cantera format
const Nprocs::Int64 = 1 # number of GPUs
const Nxp::Int64 = Nx ÷ Nprocs # make sure it is integer
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

struct constants{T, VT}
    Rg::T
    gamma::T
    C_s::T
    T_s::T
    Pr::T
    Cp::T
    CD4::VT
    Hybrid::VT
    WENO5::VT # eps, tmp1, tmp2
    UP7::VT
end

Adapt.@adapt_structure thermoProperty
Adapt.@adapt_structure reactionProperty
Adapt.@adapt_structure constants

# Run the simulation
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nGPU = MPI.Comm_size(comm)
if nGPU != Nprocs && rank == 0
    printstyled("Oops, nGPU ≠ $Nprocs\n", color=:red)
    flush(stdout)
    return
end
# set device on each MPI rank
device!(rank)
# constant parameters
const thermo = initThermo(mech) # now only NASA7
const react = initReact(mech)
const consts = constants(287.0, 1.4, 1.458e-6, 110.4, 0.72, 1004.5, 
         CuArray([2/3, 1/12]),
         CuArray([1e-8, 0.2]),
         CuArray([1e-14, 13/12, 1/6]),
         CuArray([-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420]))

@time time_step(rank, comm, thermo, consts, react)

MPI.Finalize()
