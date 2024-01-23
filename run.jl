include("solver.jl")
using PyCall
import Adapt

# load mesh info
const NG = h5read("metrics.h5", "NG")
const Nx = h5read("metrics.h5", "Nx")
const Ny = h5read("metrics.h5", "Ny")
const Nz = h5read("metrics.h5", "Nz")

# global variables, do not change name
const reaction::Bool = false
const dt::Float64 = 3e-8
const Time::Float64 = 1e-3
const step_out::Int32 = 500
const chk_out::Bool = true
const chk_compress_level = 3
const restart::String = "none"

const Nspecs::UInt8 = 5 # number of species
const Ncons::UInt8 = 5 # ρ ρu ρv ρw E 
const Nprim::UInt8 = 6 # ρ u v w p T
const mech::String = "./NN/air.yaml"
const nthreads::Tuple{Int32, Int32, Int32} = (8, 8, 4)
const nblock::Tuple{Int32, Int32, Int32} = (cld((Nx+2*NG), 8), 
                                            cld((Ny+2*NG), 8),
                                            cld((Nz+2*NG), 4))

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

thermo = initThermo(mech) # now only NASA7
consts = constants(287.0, 1.4, 1.458e-6, 110.4, 0.72, 1004.5, 
                   CuArray([-1/60, 3/20, -3/4]), 
                   CuArray([1/12, -2/3]),
                   CuArray([0.01, 0.1]),
                   CuArray([CUDA.eps(1e-16), 13/12, 1/6]),
                   CuArray([-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420]))

#initialization on CPU
function initialize(Q, Yi, consts)
    ct = pyimport("cantera")
    gas = ct.Solution(mech)
    T::Float64 = 250
    P::Float64 = 3596
    Ma::Float64 = 10
    gas.TPY = T, P, "N2:0.767 O2:0.233"
    ρ::Float64 = gas.density
    u::Float64 = Ma * gas.sound_speed

    Q[:, :, :, 1] .= ρ
    Q[:, :, :, 2] .= u
    Q[:, :, :, 3] .= 0.0
    Q[:, :, :, 4] .= 0.0
    Q[:, :, :, 5] .= P
    Q[:, :, :, 6] .= T
    for k = 1:Nz+2*NG, j = 1:Ny+2*NG, i = 1:Nx+2*NG
        Yi[i, j, k, :] .= gas.Y
    end
end

@time time_step(thermo, consts)

println("Done!")
flush(stdout)
