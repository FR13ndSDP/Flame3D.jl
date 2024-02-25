
nPoints = 100
Nspecs = 20
dt = 1e-3
mech = "../NN/CH4/drm19.yaml"
nthreads = 8

inputs = zeros(Float64, Nspecs+2, nPoints)

inputs[1, :] .= 1800
inputs[2, :] .= 101325
inputs[6, :] .= 0.5
inputs[13, :] .= 0.5

@ccall "./libchem.so".run(nPoints::Cint, Nspecs::Cint, dt::Cdouble, inputs::Ptr{Cdouble}, mech::Cstring, nthreads::Cint)::Cvoid

@show inputs[1, :]