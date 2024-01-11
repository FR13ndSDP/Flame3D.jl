include("solver.jl")
using PyCall
import Adapt

# load mesh info
const NG = h5read("metrics.h5", "metrics/NG")
const Nx = h5read("metrics.h5", "metrics/Nx")
const Ny = h5read("metrics.h5", "metrics/Ny")
const Nz = h5read("metrics.h5", "metrics/Nz")

# global variables, do not change name
const dt::Float64 = 1e-7
const Time::Float64 = 5e-4
const step_out::Int64 = 1000
const Nspecs::Int64 = 5 # number of species
const Ncons::Int64 = 5 # ρ ρu ρv ρw E 
const Nprim::Int64 = 7 # ρ u v w p T c 
const mech = "./NN/air.yaml"
const reaction::Bool = true
const nthreads::Tuple{Int64, Int64, Int64} = (8, 8, 4)
const nblock::Tuple{Int64, Int64, Int64} = (cld((Nx+2*NG), 8), 
                                            cld((Ny+2*NG), 8),
                                            cld((Nz+2*NG), 4))

struct thermoProperty{IT, RT, VT, MT, TT}
    Nspecs::IT
    Ru::RT
    mw::VT
    coeffs_sep::VT
    coeffs_lo::MT
    coeffs_hi::MT
    visc_poly::MT
    conduct_poly::MT
    binarydiff_poly::TT
end

Adapt.@adapt_structure thermoProperty

# thermo = initThermo(mech, Nspecs) # now only NASA7
@time time_step()

println("Done!")
flush(stdout)
