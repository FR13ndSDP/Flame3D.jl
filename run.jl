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
const reaction::Bool = true     # if reaction is activated
const Luxmodel::Bool = false       # if use Neural network model
const Cantera::Bool = true        # if use Cantera
const stiff::Bool = true          # if reaction is stiff
const sub_step::Int64 = 5          # reaction substep
const T_criteria::Float64 = 700.0  # reaction temperature criteria 
const dt::Float64 = 7e-8           # dt for simulation, make CFL < 1
const Time::Float64 = 1e-3         # total simulation time
const step_out::Int64 = 200        # how many steps to save result
const chk_out::Bool = false        # if checkpoint is made on save
const chk_compress_level = 3       # checkpoint file compression level 0-3, 0 for no compression
const restart::String = "none"     # restart use checkpoint, file name "chk**.h5"

const Nspecs::Int64 = 20 # number of species
const Ncons::Int64 = 5 # ρ ρu ρv ρw E 
const Nprim::Int64 = 7 # ρ u v w p T ei
const Nreacs::Int64 = 82 # number of reactions, consistent with mech
const mech::String = "./NN/CH4/drm19.yaml" # reaction mechanism file in cantera format
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
    CD6::VT
    CD4::VT
    Hybrid::VT
    WENO5::VT # eps, tmp1, tmp2
    UP7::VT
end

Adapt.@adapt_structure thermoProperty
Adapt.@adapt_structure reactionProperty
Adapt.@adapt_structure constants

#initialization on GPU
function initialize(Q, ρi, thermo)
    ct = pyimport("cantera")
    gas = ct.Solution(mech)
    T::Float64 = 300.0
    T_ignite::Float64 = 2000.0
    P::Float64 = 101325.0 * 200
    gas.TPY = T, P, "CH4:1"
    ρ::Float64 = gas.density
    gas.TPY = T_ignite, P, "CH4:1"
    ρ_ig::Float64 = gas.density
    u::Float64 = 10.0
    v::Float64 = 0.0
    w::Float64 = 0.0
    
    @cuda threads=nthreads blocks=nblock init(Q, ρi, ρ, u, v, w, P, T, T_ignite, ρ_ig, thermo)
end


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
         CuArray([-1/60, 3/20, -3/4]), 
         CuArray([1/12, -2/3]),
         CuArray([1e-6, 0.1]),
         CuArray([CUDA.eps(1e-16), 13/12, 1/6]),
         CuArray([-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420]))

@time time_step(rank, comm, thermo, consts, react)

MPI.Finalize()
