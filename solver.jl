using WriteVTK
using Lux, LuxCUDA
using LinearAlgebra, CUDA
using JLD2, JSON, HDF5

CUDA.allowscalar(false)
include("split.jl")
include("schemes.jl")
include("viscous.jl")
include("boundary.jl")
include("utils.jl")
include("div.jl")

#initialization on CPU
function initialize(U, ρi)
    ct = pyimport("cantera")
    gas = ct.Solution(mech)
    T::Float64 = 350
    P::Float64 = 3596
    gas.TPY = T, P, "N2:0.767 O2:0.233"
    ρ::Float64 = P/(287*T)
    c::Float64 = sqrt(1.4 * P / ρ)
    u::Float64 = 10 * c

    U[:, :, :, 1] .= ρ
    U[:, :, :, 2] .= ρ * u
    U[:, :, :, 3] .= 0.0
    U[:, :, :, 4] .= 0.0
    U[:, :, :, 5] .= P/(1.4-1) + 0.5 * ρ * u^2
    for k ∈ 1:Nz+2*NG, j ∈ 1:Ny+2*NG, i ∈ 1:Nx+2*NG
        ρi[i, j, k, :] .= gas.Y .* ρ
    end
end

function correction(U, ρi, NG, Nx, Ny, Nz)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z
    if i > Nx+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    @inbounds ρ = U[i, j, k, 1]
    ∑ρ = 0
    for n = 1:Nspecs
        @inbounds ρn = ρi[i, j, k, n]
        if ρn < 0
            ρi[i, j, k, n] = 0
        end
        @inbounds ∑ρ += ρi[i, j, k, n]
    end
    for n = 1:Nspecs
        @inbounds ρi[i, j, k, n] *= ρ/∑ρ
    end
end

function flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ)
    # local constants
    gamma::Float64 = 1.4
    Ru::Float64 = 8.31446
    eos_m::Float64 = 28.97e-3
    Rg::Float64 = Ru/eos_m
    Pr::Float64 = 0.72
    Cv::Float64 = Rg/(gamma-1)
    Cp::Float64 = gamma * Cv
    C_s::Float64 = 1.458e-6
    T_s::Float64 = 110.4

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock fluxSplit(Q, U, Fp, Fm, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz)
    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock WENO_x(Fx, ϕ, Fp, Fm, NG, Nx, Ny, Nz, Ncons)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock fluxSplit(Q, U, Fp, Fm, Nx, Ny, Nz, NG, dηdx, dηdy, dηdz)
    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock WENO_y(Fy, ϕ, Fp, Fm, NG, Nx, Ny, Nz, Ncons)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock fluxSplit(Q, U, Fp, Fm, Nx, Ny, Nz, NG, dζdx, dζdy, dζdz)
    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock WENO_z(Fz, ϕ, Fp, Fm, NG, Nx, Ny, Nz, Ncons)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock viscousFlux(Fv_x, Fv_y, Fv_z, Q, NG, Nx, Ny, Nz, Pr, Cp, C_s, T_s, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock div(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, NG, Nx, Ny, Nz, J)

end

function specAdvance(U, ρi, Q, Fp_i, Fm_i, Fx_i, Fy_i, Fz_i, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ)
    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock c2Prim(U, Q, Nx, Ny, Nz, NG, 1.4, 287)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock shockSensor(ϕ, Q, Nx, Ny, Nz, NG)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock split(ρi, Q, U, Fp_i, Fm_i, dξdx, dξdy, dξdz, Nx, Ny, Nz, NG)
    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock NND_x(Fx_i, Fp_i, Fm_i, NG, Nx, Ny, Nz, Nspecs)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock split(ρi, Q, U, Fp_i, Fm_i, dηdx, dηdy, dηdz, Nx, Ny, Nz, NG)
    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock NND_y(Fy_i, Fp_i, Fm_i, NG, Nx, Ny, Nz, Nspecs)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock split(ρi, Q, U, Fp_i, Fm_i, dζdx, dζdy, dζdz, Nx, Ny, Nz, NG)
    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock NND_z(Fz_i, Fp_i, Fm_i, NG, Nx, Ny, Nz, Nspecs)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock divSpecs(ρi, Fx_i, Fy_i, Fz_i, dt, NG, Nx, Ny, Nz, J)
end

# Collect input
function pre_input(inputs, inputs_norm, Q, ρi, lambda, inputs_mean, inputs_std, Nx, Ny, Nz, NG)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nx || j > Ny || k > Nz
        return
    end

    @inbounds inputs[1, i + Nx*(j-1 + Ny*(k-1))] = Q[i+NG, j+NG, k+NG, 6] # T
    @inbounds inputs[2, i + Nx*(j-1 + Ny*(k-1))] = Q[i+NG, j+NG, k+NG, 5] # p

    for n = 3:Nspecs+2
        @inbounds Yi = CUDA.max(ρi[i+NG, j+NG, k+NG, n-2]/Q[i+NG, j+NG, k+NG, 1], 0)
        @inbounds inputs[n, i + Nx*(j-1 + Ny*(k-1))] = (Yi^lambda - 1) / lambda
    end

    for n = 1:Nspecs+2
        @inbounds inputs_norm[n, i + Nx*(j-1 + Ny*(k-1))] = (inputs[n, i + Nx*(j-1 + Ny*(k-1))] - inputs_mean[n]) / inputs_std[n]
    end
    return
end

# Parse prediction
function post_predict(yt_pred, inputs, U, Q, ρi, dt, lambda, Nx, Ny, Nz, NG)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nx || j > Ny || k > Nz
        return
    end

    T = Q[i+NG, j+NG, k+NG, 6]
    # only T > 2000 K calculate reaction
    if T > 2000
        @inbounds T += yt_pred[Nspecs+1, i + Nx*(j-1 + Ny*(k-1))] * dt
        @inbounds U[i+NG, j+NG, k+NG, 5] += (Q[i+NG, j+NG, k+NG, 1] * 287 * T - Q[i+NG, j+NG, k+NG, 5])/0.4
        for n = 1:Nspecs
            @inbounds Yi = (lambda * (yt_pred[n, i + Nx*(j-1 + Ny*(k-1))] * dt + inputs[n+2, i + Nx*(j-1 + Ny*(k-1))]) + 1) ^ (1/lambda)
            @inbounds ρi[i+NG, j+NG, k+NG, n] = Yi * Q[i+NG, j+NG, k+NG, 1]
        end
    end
    return
end

# Zero GPU allocation
function evalModel(Y1, Y2, output, w1, w2, w3, b1, b2, b3, input)
    mul!(Y1, w1, input)
    Y1 .+= b1
    @. Y1 = gelu(Y1)

    mul!(Y2, w2, Y1)
    Y2 .+= b2
    @. Y2 = gelu(Y2)

    mul!(output, w3, Y2)
    output .+= b3

    return
end

function time_step()
    Nx_tot = Nx+2*NG
    Ny_tot = Ny+2*NG
    Nz_tot = Nz+2*NG

    U_h = zeros(Float64, Nx+2*NG, Ny+2*NG, Nz+2*NG, Ncons)
    ρi_h = zeros(Float64, Nx+2*NG, Ny+2*NG, Nz+2*NG, Nspecs)
    ϕ_h = zeros(Float64, Nx+2*NG, Ny+2*NG, Nz+2*NG) # shock sensor

    initialize(U_h, ρi_h)
    
    # load mesh metrics
    dξdx_h = h5read("metrics.h5", "metrics/dξdx")
    dξdy_h = h5read("metrics.h5", "metrics/dξdy")
    dξdz_h = h5read("metrics.h5", "metrics/dξdz")
    dηdx_h = h5read("metrics.h5", "metrics/dηdx")
    dηdy_h = h5read("metrics.h5", "metrics/dηdy")
    dηdz_h = h5read("metrics.h5", "metrics/dηdz")
    dζdx_h = h5read("metrics.h5", "metrics/dζdx")
    dζdy_h = h5read("metrics.h5", "metrics/dζdy")
    dζdz_h = h5read("metrics.h5", "metrics/dζdz")

    J_h = h5read("metrics.h5", "metrics/J") 
    x_h = h5read("metrics.h5", "metrics/x") 
    y_h = h5read("metrics.h5", "metrics/y") 
    z_h = h5read("metrics.h5", "metrics/z") 

    # move to device memory
    U = CuArray(U_h)
    ρi = CuArray(ρi_h)
    ϕ = CuArray(ϕ_h)
    dξdx = CuArray(dξdx_h)
    dξdy = CuArray(dξdy_h)
    dξdz = CuArray(dξdz_h)
    dηdx = CuArray(dηdx_h)
    dηdy = CuArray(dηdy_h)
    dηdz = CuArray(dηdz_h)
    dζdx = CuArray(dζdx_h)
    dζdy = CuArray(dζdy_h)
    dζdz = CuArray(dζdz_h)
    J = CuArray(J_h)

    # allocate on device
    Un = copy(U)
    ρn = copy(ρi)
    Q  =   CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nprim)
    Fp =   CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fm =   CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fx =   CUDA.zeros(Float64, Nx-1, Ny-2, Nz-2, Ncons)
    Fy =   CUDA.zeros(Float64, Nx-2, Ny-1, Nz-2, Ncons)
    Fz =   CUDA.zeros(Float64, Nx-2, Ny-2, Nz-1, Ncons)
    Fv_x = CUDA.zeros(Float64, Nx_tot-4, Ny_tot-4, Nz_tot-4, 4)
    Fv_y = CUDA.zeros(Float64, Nx_tot-4, Ny_tot-4, Nz_tot-4, 4)
    Fv_z = CUDA.zeros(Float64, Nx_tot-4, Ny_tot-4, Nz_tot-4, 4)
    Fp_i = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
    Fm_i = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
    Fx_i = CUDA.zeros(Float64, Nx-1, Ny-2, Nz-2, Nspecs)
    Fy_i = CUDA.zeros(Float64, Nx-2, Ny-1, Nz-2, Nspecs)
    Fz_i = CUDA.zeros(Float64, Nx-2, Ny-2, Nz-1, Nspecs)

    # fill boundary
    fillGhost(U, NG, Nx, Ny, Nz)
    fillSpec(ρi, U, NG, Nx, Ny, Nz)

    if reaction
        @load "./NN/luxmodel.jld2" model ps st

        ps = ps |> gpu_device()

        w1 = ps[1].weight
        b1 = ps[1].bias
        w2 = ps[2].weight
        b2 = ps[2].bias
        w3 = ps[3].weight
        b3 = ps[3].bias

        Y1 = CUDA.ones(Float32, 128, Nx*Ny*Nz)
        Y2 = CUDA.ones(Float32, 64, Nx*Ny*Nz)
        yt_pred = CUDA.ones(Float32, Nspecs+1, Nx*Ny*Nz)

        j = JSON.parsefile("./NN/norm.json")
        lambda = j["lambda"]
        inputs_mean = CuArray(convert(Vector{Float32}, j["inputs_mean"]))
        inputs_std =  CuArray(convert(Vector{Float32}, j["inputs_std"]))
        labels_mean = CuArray(convert(Vector{Float32}, j["labels_mean"]))
        labels_std =  CuArray(convert(Vector{Float32}, j["labels_std"]))

        inputs = CUDA.zeros(Float32, Nspecs+2, Nx*Ny*Nz)
        inputs_norm = CUDA.zeros(Float32, Nspecs+2, Nx*Ny*Nz)

        dt2 = dt/2
    end

    for tt ∈ 1:ceil(Int, Time/dt)
        if tt % 10 == 0
            printstyled("Step: ", color=:cyan)
            print("$tt")
            printstyled("\tTime: ", color=:blue)
            println("$(tt*dt)")
            if any(isnan, U)
                printstyled("Oops, NaN detected\n", color=:red)
                return
            end
            flush(stdout)
        end

        if reaction
            # Reaction Step
            @cuda threads=nthreads blocks=nblock c2Prim(U, Q, Nx, Ny, Nz, NG, 1.4, 287)
            @cuda threads=nthreads blocks=nblock pre_input(inputs, inputs_norm, Q, ρi, lambda, inputs_mean, inputs_std, Nx, Ny, Nz, NG)
            evalModel(Y1, Y2, yt_pred, w1, w2, w3, b1, b2, b3, inputs_norm)
            @. yt_pred = yt_pred * labels_std + labels_mean
            @cuda threads=nthreads blocks=nblock post_predict(yt_pred, inputs, U, Q, ρi, dt2, lambda, Nx, Ny, Nz, NG)
            @cuda threads=nthreads blocks=nblock correction(U, ρi, NG, Nx, Ny, Nz)
            fillGhost(U, NG, Nx, Ny, Nz)
            fillSpec(ρi, U, NG, Nx, Ny, Nz)
        end

        # RK3-1
        @cuda threads=nthreads blocks=nblock copyOld(Un, U, Nx, Ny, Nz, NG, Ncons)
        @cuda threads=nthreads blocks=nblock copyOld(ρn, ρi, Nx, Ny, Nz, NG, Nspecs)

        specAdvance(U, ρi, Q, Fp_i, Fm_i, Fx_i, Fy_i, Fz_i, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ)
        flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ)
        fillGhost(U, NG, Nx, Ny, Nz)
        fillSpec(ρi, U, NG, Nx, Ny, Nz)

        # RK3-2
        specAdvance(U, ρi, Q, Fp_i, Fm_i, Fx_i, Fy_i, Fz_i, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ)
        flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ)
        @cuda threads=nthreads blocks=nblock linComb(U, Un, Nx, Ny, Nz, NG, Ncons, 0.25, 0.75)
        @cuda threads=nthreads blocks=nblock linComb(ρi, ρn, Nx, Ny, Nz, NG, Nspecs, 0.25, 0.75)
        fillGhost(U, NG, Nx, Ny, Nz)
        fillSpec(ρi, U, NG, Nx, Ny, Nz)

        # RK3-3
        specAdvance(U, ρi, Q, Fp_i, Fm_i, Fx_i, Fy_i, Fz_i, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ)
        flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ)
        @cuda threads=nthreads blocks=nblock linComb(U, Un, Nx, Ny, Nz, NG, Ncons, 2/3, 1/3)
        @cuda threads=nthreads blocks=nblock linComb(ρi, ρn, Nx, Ny, Nz, NG, Nspecs, 2/3, 1/3)
        @cuda threads=nthreads blocks=nblock correction(U, ρi, NG, Nx, Ny, Nz)
        fillGhost(U, NG, Nx, Ny, Nz)
        fillSpec(ρi, U, NG, Nx, Ny, Nz)

        if reaction
            # Reaction Step
            @cuda threads=nthreads blocks=nblock c2Prim(U, Q, Nx, Ny, Nz, NG, 1.4, 287)
            @cuda threads=nthreads blocks=nblock pre_input(inputs, inputs_norm, Q, ρi, lambda, inputs_mean, inputs_std, Nx, Ny, Nz, NG)
            evalModel(Y1, Y2, yt_pred, w1, w2, w3, b1, b2, b3, inputs_norm)
            @. yt_pred = yt_pred * labels_std + labels_mean
            @cuda threads=nthreads blocks=nblock post_predict(yt_pred, inputs, U, Q, ρi, dt2, lambda, Nx, Ny, Nz, NG)
            @cuda threads=nthreads blocks=nblock correction(U, ρi, NG, Nx, Ny, Nz)
            fillGhost(U, NG, Nx, Ny, Nz)
            fillSpec(ρi, U, NG, Nx, Ny, Nz)
        end

        if tt % step_out == 0 || tt == ceil(Int, Time/dt)
            copyto!(U_h, U)
            copyto!(ρi_h, ρi)
            copyto!(ϕ_h, ϕ)
            fname::String = string("plt", tt)

            rho = U_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 1]
            u =   U_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 2]./rho
            v =   U_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 3]./rho
            w =   U_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 4]./rho
            p = @. (U_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 5] - 0.5*rho*(u^2+v^2+w^2)) * 0.4
            T = @. p/(287.0 * rho)
        
            YO = ρi_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 1]./rho
            YO2 = ρi_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 2]./rho
            YN = ρi_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 3]./rho
            YNO = ρi_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 4]./rho
            YN2 = ρi_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 5]./rho

            @views ϕ_ng = ϕ_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG]
            @views x_ng = x_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG]
            @views y_ng = y_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG]
            @views z_ng = z_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG]

            vtk_grid(fname, x_ng, y_ng, z_ng) do vtk
                vtk["rho"] = rho
                vtk["u"] = u
                vtk["v"] = v
                vtk["w"] = w
                vtk["p"] = p
                vtk["T"] = T
                vtk["phi"] = ϕ_ng
                vtk["YO"] = YO
                vtk["YO2"] = YO2
                vtk["YN"] = YN
                vtk["YNO"] = YNO
                vtk["YN2"] = YN2
            end 
        end
    end
    return
end
