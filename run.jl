include("solver.jl")
using PyCall

# load mesh metrics
@load "metrics.jld2" NG Nx Ny Nz dξdx dξdy dξdz dηdx dηdy dηdz dζdx dζdy dζdz J x y z

# global variables, do not change name
const dt::Float64 = 1e-7
const Time::Float64 = 5e-4
const Ncons::Int64 = 5 # ρ ρu ρv ρw E 
const Nprim::Int64 = 7 # ρ u v w p T c 
const mech = "./air.yaml"

U = zeros(Float64, Nx+2*NG, Ny+2*NG, Nz+2*NG, Ncons)

#initialization on CPU
function initialize(U, mech)
    ct = pyimport("cantera")
    gas = ct.Solution(mech)
    T::Float64 = 350
    P::Float64 = 3596
    gas.TPY = T, P, "N2:0.767 O2:0.233"
    ρ::Float64 = gas.density
    c::Float64 = sqrt(1.4 * P / ρ)
    u::Float64 = 10 * c

    U[:, :, :, 1] .= ρ
    U[:, :, :, 2] .= ρ * u
    U[:, :, :, 3] .= 0.0
    U[:, :, :, 4] .= 0.0
    U[:, :, :, 5] .= P/(1.4-1) + 0.5 * ρ * u^2
end

initialize(U, mech)

U_d = CuArray(U)
dξdx_d = CuArray(dξdx)
dξdy_d = CuArray(dξdy)
dξdz_d = CuArray(dξdz)
dηdx_d = CuArray(dηdx)
dηdy_d = CuArray(dηdy)
dηdz_d = CuArray(dηdz)
dζdx_d = CuArray(dζdx)
dζdy_d = CuArray(dζdy)
dζdz_d = CuArray(dζdz)
J_d = CuArray(J)

@time time_step(U_d, dξdx_d, dξdy_d, dξdz_d, dηdx_d, dηdy_d, dηdz_d, dζdx_d, dζdy_d, dζdz_d, J_d, Nx, Ny, Nz, NG, dt)
copyto!(U, U_d)

rho = U[:, :, :, 1]
u =   U[:, :, :, 2]./rho
v =   U[:, :, :, 3]./rho
w =   U[:, :, :, 4]./rho
p = @. (U[:, :, :, 5] - 0.5*rho*(u^2+v^2+w^2)) * 0.4
T = @. p/(287.0 * rho)

vtk_grid("result.vts", x, y, z) do vtk
    vtk["rho"] = rho
    vtk["u"] = u
    vtk["v"] = v
    vtk["w"] = w
    vtk["p"] = p
    vtk["T"] = T
end 
