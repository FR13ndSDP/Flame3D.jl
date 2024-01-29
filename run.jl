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
const reaction::Bool = true        # if reaction is activated
const Luxmodel::Bool = false       # if use Neural network model
const Cantera::Bool = false        # if use Cantera
const stiff::Bool = false          # if reaction is stiff
const dt::Float64 = 5e-8           # dt for simulation, make CFL < 1
const Time::Float64 = 5e-6         # total simulation time
const step_out::Int64 = 500        # how many steps to save result
const chk_out::Bool = true         # if checkpoint is made on save
const chk_compress_level = 3       # checkpoint file compression level 0-3, 0 for no compression
const restart::String = "none"     # restart use checkpoint, file name "chk**.h5"

const Nspecs::Int64 = 5 # number of species
const Ncons::Int64 = 5 # ρ ρu ρv ρw E 
const Nprim::Int64 = 7 # ρ u v w p T ei
const mech::String = "./NN/Air/air.yaml" # reaction mechanism file in cantera format
const Nprocs::Int64 = 1 # number of GPUs
const Nxp::Int64 = Nx ÷ Nprocs # make sure it is integer
const nthreads::Tuple{Int32, Int32, Int32} = (4, 8, 8)
const nblock::Tuple{Int32, Int32, Int32} = (cld((Nxp+2*NG), 4), 
                                            cld((Ny+2*NG), 8),
                                            cld((Nz+2*NG), 8))

struct thermoProperty{IT, RT, VT, MT, TT}
    Nspecs::IT
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
Adapt.@adapt_structure constants

#initialization on CPU
function initialize(Q, ρi, consts)
    ct = pyimport("cantera")
    gas = ct.Solution(mech)
    T::Float64 = 1000
    P::Float64 = 3596
    Ma::Float64 = 0.1
    gas.TPY = T, P, "N2:77 O2:23"
    ρ::Float64 = gas.density
    u::Float64 = Ma * gas.sound_speed
    ei::Float64 = 9576.368742188555 # get from InternalEnergy

    Q[:, :, :, 1] .= ρ
    Q[:, :, :, 2] .= u
    Q[:, :, :, 3] .= 0.0
    Q[:, :, :, 4] .= 0.0
    Q[:, :, :, 5] .= P
    Q[:, :, :, 6] .= T
    Q[:, :, :, 7] .= ei
    for k = 1:Nz+2*NG, j = 1:Ny+2*NG, i = 1:Nxp+2*NG
        ρi[i, j, k, :] .= gas.Y * ρ
    end
end

MPI.Init()

@time time_step()

MPI.Finalize()
