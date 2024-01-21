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
include("reaction.jl")

function correction(U, ρi)
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
    # for n = 1:Nspecs
    #     @inbounds ρi[i, j, k, n] *= ρ/∑ρ
    # end
    ρi[i, j, k, Nspecs] += ρ - ∑ρ
    return
end

function flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, λ, μ, Fh, consts)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock fluxSplit(Q, U, Fp, Fm, dξdx, dξdy, dξdz)
    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock WENO_x(Fx, ϕ, Fp, Fm, Ncons, consts)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock fluxSplit(Q, U, Fp, Fm, dηdx, dηdy, dηdz)
    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock WENO_y(Fy, ϕ, Fp, Fm, Ncons, consts)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock fluxSplit(Q, U, Fp, Fm, dζdx, dζdy, dζdz)
    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock WENO_z(Fz, ϕ, Fp, Fm, Ncons, consts)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock viscousFlux(Fv_x, Fv_y, Fv_z, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, λ, μ, Fh, consts)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock div(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J, consts)
end

function specAdvance(U, ρi, Q, Yi, Fp_i, Fm_i, Fx_i, Fy_i, Fz_i, Fd_x, Fd_y, Fd_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, D, Fh, thermo, consts)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock split(ρi, Q, U, Fp_i, Fm_i, dξdx, dξdy, dξdz)
    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock WENO_x(Fx_i, ϕ, Fp_i, Fm_i, Nspecs, consts)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock split(ρi, Q, U, Fp_i, Fm_i, dηdx, dηdy, dηdz)
    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock WENO_y(Fy_i, ϕ, Fp_i, Fm_i, Nspecs, consts)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock split(ρi, Q, U, Fp_i, Fm_i, dζdx, dζdy, dζdz)
    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock WENO_z(Fz_i, ϕ, Fp_i, Fm_i, Nspecs, consts)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock specViscousFlux(Fd_x, Fd_y, Fd_z, Q, Yi, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, D, Fh, thermo, consts)

    @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock divSpecs(ρi, Fx_i, Fy_i, Fz_i, Fd_x, Fd_y, Fd_z, dt, J, consts)
end

# Collect input
function pre_input(inputs, inputs_norm, Q, Y, lambda, inputs_mean, inputs_std)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nx || j > Ny || k > Nz
        return
    end

    @inbounds inputs[1, i + Nx*(j-1 + Ny*(k-1))] = Q[i+NG, j+NG, k+NG, 6] # T
    @inbounds inputs[2, i + Nx*(j-1 + Ny*(k-1))] = Q[i+NG, j+NG, k+NG, 5] # p

    for n = 3:Nspecs+2
        @inbounds Yi = Y[i+NG, j+NG, k+NG, n-2]
        @inbounds inputs[n, i + Nx*(j-1 + Ny*(k-1))] = (Yi^lambda - 1) / lambda
    end

    for n = 1:Nspecs+2
        @inbounds inputs_norm[n, i + Nx*(j-1 + Ny*(k-1))] = (inputs[n, i + Nx*(j-1 + Ny*(k-1))] - inputs_mean[n]) / inputs_std[n]
    end
    return
end

# Parse prediction
function post_predict(yt_pred, inputs, U, Q, ρi, dt, lambda, thermo)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nx || j > Ny || k > Nz
        return
    end

    T = Q[i+NG, j+NG, k+NG, 6]
    P = Q[i+NG, j+NG, k+NG, 5]
    @inbounds rhoi = @view ρi[i+NG, j+NG, k+NG, :]

    # only T > 2000 K calculate reaction
    if T > 3000 && P < 10132.5
        @inbounds T1 = T + yt_pred[Nspecs+1, i + Nx*(j-1 + Ny*(k-1))] * dt
        @inbounds U[i+NG, j+NG, k+NG, 5] += InternalEnergy(T1, rhoi, thermo) - InternalEnergy(T, rhoi, thermo)
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

function time_step(thermo, consts)
    Nx_tot = Nx+2*NG
    Ny_tot = Ny+2*NG
    Nz_tot = Nz+2*NG

    if restart[1:3] == "chk"
        printstyled("Restart\n", color=:red)
        U_h = h5read(restart, "U_h")
        ρi_h = h5read(restart, "ρi_h")
    else
        U_h = zeros(Float64, Nx+2*NG, Ny+2*NG, Nz+2*NG, Ncons)
        ρi_h = zeros(Float64, Nx+2*NG, Ny+2*NG, Nz+2*NG, Nspecs)
        initialize(U_h, ρi_h, consts)
    end

    ϕ_h = zeros(Float32, Nx+2*NG, Ny+2*NG, Nz+2*NG) # shock sensor, single precision
    
    # load mesh metrics
    dξdx_h = h5read("metrics.h5", "dξdx")
    dξdy_h = h5read("metrics.h5", "dξdy")
    dξdz_h = h5read("metrics.h5", "dξdz")
    dηdx_h = h5read("metrics.h5", "dηdx")
    dηdy_h = h5read("metrics.h5", "dηdy")
    dηdz_h = h5read("metrics.h5", "dηdz")
    dζdx_h = h5read("metrics.h5", "dζdx")
    dζdy_h = h5read("metrics.h5", "dζdy")
    dζdz_h = h5read("metrics.h5", "dζdz")

    J_h = h5read("metrics.h5", "J") 
    x_h = h5read("metrics.h5", "x") 
    y_h = h5read("metrics.h5", "y") 
    z_h = h5read("metrics.h5", "z")

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
    Fx_i = CUDA.zeros(Float64, Nx-1, Ny-2, Nz-2, Nspecs) # species advection
    Fy_i = CUDA.zeros(Float64, Nx-2, Ny-1, Nz-2, Nspecs) # species advection
    Fz_i = CUDA.zeros(Float64, Nx-2, Ny-2, Nz-1, Nspecs) # species advection
    Fd_x = CUDA.zeros(Float64, Nx_tot-4, Ny_tot-4, Nz_tot-4, Nspecs) # species diffusion
    Fd_y = CUDA.zeros(Float64, Nx_tot-4, Ny_tot-4, Nz_tot-4, Nspecs) # species diffusion
    Fd_z = CUDA.zeros(Float64, Nx_tot-4, Ny_tot-4, Nz_tot-4, Nspecs) # species diffusion
    Fh = CUDA.zeros(Float64, Nx_tot-4, Ny_tot-4, Nz_tot-4, 3) # enthalpy diffusion

    μ = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    λ = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    D = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)

    μi = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs) # tmp for both μ and λ
    Xi = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
    Yi = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
    Dij = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs*Nspecs) # tmp for N*N matricies
    
    # fill boundary
    fillGhost(U)
    fillSpec(ρi, U)
    @cuda threads=nthreads blocks=nblock c2Prim(U, Q, ρi, Yi, thermo)

    if reaction
        @load "./NN/luxmodel.jld2" model ps st

        ps = ps |> gpu_device()

        w1 = ps[1].weight
        b1 = ps[1].bias
        w2 = ps[2].weight
        b2 = ps[2].bias
        w3 = ps[3].weight
        b3 = ps[3].bias

        Y1 = CUDA.ones(Float32, 64, Nx*Ny*Nz)
        Y2 = CUDA.ones(Float32, 256, Nx*Ny*Nz)
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

    for tt = 1:ceil(Int, Time/dt)
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
            @cuda threads=nthreads blocks=nblock pre_input(inputs, inputs_norm, Q, Yi, lambda, inputs_mean, inputs_std)
            evalModel(Y1, Y2, yt_pred, w1, w2, w3, b1, b2, b3, inputs_norm)
            @. yt_pred = yt_pred * labels_std + labels_mean
            @cuda threads=nthreads blocks=nblock post_predict(yt_pred, inputs, U, Q, ρi, dt2, lambda, thermo)
            @cuda threads=nthreads blocks=nblock correction(U, ρi)
            fillGhost(U)
            fillSpec(ρi, U)
            @cuda threads=nthreads blocks=nblock c2Prim(U, Q, ρi, Yi, thermo)
        end

        # RK3-1
        @cuda threads=nthreads blocks=nblock copyOld(Un, U, Ncons)
        @cuda threads=nthreads blocks=nblock copyOld(ρn, ρi, Nspecs)

        @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock mixture(Q, Yi, Xi, μi, Dij, λ, μ, D, thermo)
        @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock shockSensor(ϕ, Q)
        specAdvance(U, ρi, Q, Yi, Fp_i, Fm_i, Fx_i, Fy_i, Fz_i, Fd_x, Fd_y, Fd_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, D, Fh, thermo, consts)
        flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, λ, μ, Fh, consts)
        fillGhost(U)
        fillSpec(ρi,U)
        @cuda threads=nthreads blocks=nblock c2Prim(U, Q, ρi, Yi, thermo)

        # RK3-2
        @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock mixture(Q, Yi, Xi, μi, Dij, λ, μ, D, thermo)
        @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock shockSensor(ϕ, Q)
        specAdvance(U, ρi, Q, Yi, Fp_i, Fm_i, Fx_i, Fy_i, Fz_i, Fd_x, Fd_y, Fd_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, D, Fh, thermo, consts)
        flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, λ, μ, Fh, consts)
        @cuda threads=nthreads blocks=nblock linComb(U, Un, Ncons, 0.25, 0.75)
        @cuda threads=nthreads blocks=nblock linComb(ρi, ρn, Nspecs, 0.25, 0.75)
        fillGhost(U)
        fillSpec(ρi, U)
        @cuda threads=nthreads blocks=nblock c2Prim(U, Q, ρi, Yi, thermo)

        # RK3-3
        @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock mixture(Q, Yi, Xi, μi, Dij, λ, μ, D, thermo)
        @cuda maxregs=255 fastmath=true threads=nthreads blocks=nblock shockSensor(ϕ, Q)
        specAdvance(U, ρi, Q, Yi, Fp_i, Fm_i, Fx_i, Fy_i, Fz_i, Fd_x, Fd_y, Fd_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, D, Fh, thermo, consts)
        flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, λ, μ, Fh, consts)
        @cuda threads=nthreads blocks=nblock linComb(U, Un, Ncons, 2/3, 1/3)
        @cuda threads=nthreads blocks=nblock linComb(ρi, ρn, Nspecs, 2/3, 1/3)
        @cuda threads=nthreads blocks=nblock correction(U, ρi)
        fillGhost(U)
        fillSpec(ρi, U)
        @cuda threads=nthreads blocks=nblock c2Prim(U, Q, ρi, Yi, thermo)

        if reaction
            # Reaction Step
            @cuda threads=nthreads blocks=nblock pre_input(inputs, inputs_norm, Q, Yi, lambda, inputs_mean, inputs_std)
            evalModel(Y1, Y2, yt_pred, w1, w2, w3, b1, b2, b3, inputs_norm)
            @. yt_pred = yt_pred * labels_std + labels_mean
            @cuda threads=nthreads blocks=nblock post_predict(yt_pred, inputs, U, Q, ρi, dt2, lambda, thermo)
            @cuda threads=nthreads blocks=nblock correction(U, ρi)
            fillGhost(U)
            fillSpec(ρi, U)
            @cuda threads=nthreads blocks=nblock c2Prim(U, Q, ρi, Yi, thermo)
        end

        if tt % step_out == 0 || tt == ceil(Int, Time/dt)
            Q_h = zeros(Float64, Nx+2*NG, Ny+2*NG, Nz+2*NG, Nprim)
            Yi_h = zeros(Float64, Nx+2*NG, Ny+2*NG, Nz+2*NG, Nspecs)
            μ_h = zeros(Float64, Nx+2*NG, Ny+2*NG, Nz+2*NG)
            λ_h = zeros(Float64, Nx+2*NG, Ny+2*NG, Nz+2*NG)
            copyto!(Q_h, Q)
            copyto!(Yi_h, Yi)
            copyto!(ϕ_h, ϕ)
            copyto!(μ_h, μ)
            copyto!(λ_h, λ)

            # visualization file, in Float32
            fname::String = string("plt", tt)

            rho = convert(Array{Float32, 3}, @view Q_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 1])
            u =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 2])
            v =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 3])
            w =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 4])
            p =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 5])
            T =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 6])
        
            YO  = convert(Array{Float32, 3}, @view Yi_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 1])
            YO2 = convert(Array{Float32, 3}, @view Yi_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 2])
            YN  = convert(Array{Float32, 3}, @view Yi_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 3])
            YNO = convert(Array{Float32, 3}, @view Yi_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 4])
            YN2 = convert(Array{Float32, 3}, @view Yi_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 5])

            ϕ_ng = @view ϕ_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG]
            x_ng = convert(Array{Float32, 3}, @view x_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG])
            y_ng = convert(Array{Float32, 3}, @view y_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG])
            z_ng = convert(Array{Float32, 3}, @view z_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG])

            μ_ng = convert(Array{Float32, 3}, @view μ_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG])
            λ_ng = convert(Array{Float32, 3}, @view λ_h[1+NG:Nx+NG, 1+NG:Ny+NG, 1+NG:Nz+NG])

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
                vtk["mu"] = μ_ng
                vtk["lambda"] = λ_ng
            end 

            # restart file, in Float64
            if chk_out
                copyto!(U_h, U)
                copyto!(ρi_h, ρi)
                chkname::String = string("chk", tt, ".h5")
                h5open(chkname, "w") do file
                    file["U_h", compress=chk_compress_level] = U_h
                    file["ρi_h", compress=chk_compress_level] = ρi_h
                end
            end
            
            # release memory
            Q_h = nothing
            Yi_h = nothing
        end
    end
    return
end
