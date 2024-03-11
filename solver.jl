using MPI
using WriteVTK
using Lux, LuxCUDA
using LinearAlgebra, StaticArrays, CUDA
using CUDA:i32
using JLD2, JSON, HDF5

CUDA.allowscalar(false)
LinearAlgebra.BLAS.set_num_threads(1)

include("split.jl")
include("schemes.jl")
include("viscous.jl")
include("boundary.jl")
include("utils.jl")
include("div.jl")
include("reactions.jl")
include("thermo.jl")
include("mpi.jl")

# TODO: implement charateristic flux splitting
function flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, λ, μ, Fhx, Fhy, Fhz, consts, tag)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit(Q, Fp, Fm, dξdx, dξdy, dξdz, tag)
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock WENO_x(Fx, ϕ, Fp, Fm, Ncons, consts, tag)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit(Q, Fp, Fm, dηdx, dηdy, dηdz, tag)
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock WENO_y(Fy, ϕ, Fp, Fm, Ncons, consts, tag)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit(Q, Fp, Fm, dζdx, dζdy, dζdz, tag)
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock WENO_z(Fz, ϕ, Fp, Fm, Ncons, consts, tag)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock viscousFlux_x(Fv_x, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, λ, μ, Fhx, tag)
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock viscousFlux_y(Fv_y, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, λ, μ, Fhy, tag)
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock viscousFlux_z(Fv_z, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, λ, μ, Fhz, tag)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock div(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J, tag)
end

function specAdvance(ρi, Q, Yi, Fp_i, Fm_i, Fx_i, Fy_i, Fz_i, Fd_x, Fd_y, Fd_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, D, Fhx, Fhy, Fhz, thermo, consts, tag)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock split(ρi, Q, Fp_i, Fm_i, dξdx, dξdy, dξdz, tag)
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock WENO_x(Fx_i, ϕ, Fp_i, Fm_i, Nspecs, consts, tag)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock split(ρi, Q, Fp_i, Fm_i, dηdx, dηdy, dηdz, tag)
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock WENO_y(Fy_i, ϕ, Fp_i, Fm_i, Nspecs, consts, tag)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock split(ρi, Q, Fp_i, Fm_i, dζdx, dζdy, dζdz, tag)
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock WENO_z(Fz_i, ϕ, Fp_i, Fm_i, Nspecs, consts, tag)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock specViscousFlux_x(Fd_x, Q, Yi, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, D, Fhx, thermo, tag)
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock specViscousFlux_y(Fd_y, Q, Yi, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, D, Fhy, thermo, tag)
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock specViscousFlux_z(Fd_z, Q, Yi, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, D, Fhz, thermo, tag)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock divSpecs(ρi, Fx_i, Fy_i, Fz_i, Fd_x, Fd_y, Fd_z, dt, J, tag)
end

function time_step(rank, comm, thermo, consts, react)
    Nx_tot = Nxp+2*NG
    Ny_tot = Ny+2*NG
    Nz_tot = Nz+2*NG

    # global indices
    lo = rank*Nxp+1
    hi = (rank+1)*Nxp+2*NG
    # lo_ng = rank*Nxp+NG+1
    # hi_ng = (rank+1)*Nxp+NG

    if restart[end-2:end] == ".h5"
        if rank == 0
            printstyled("Restart\n", color=:yellow)
        end
        fid = h5open(restart, "r", comm)
        Q_h = fid["Q_h"][:, :, :, :, rank+1]
        ρi_h = fid["ρi_h"][:, :, :, :, rank+1]
        close(fid)

        Q  =   CuArray(Q_h)
        ρi =   CuArray(ρi_h)
    else
        Q_h = zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nprim)
        ρi_h = zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
        Q = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nprim)
        ρi =CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
        initialize(Q, ρi, thermo)

        copyto!(Q_h, Q)
        copyto!(ρi_h, ρi)
    end
    
    ϕ_h = zeros(Float64, Nx_tot, Ny_tot, Nz_tot) # shock sensor

    # load mesh metrics
    fid = h5open("metrics.h5", "r", comm)
    dξdx_h = fid["dξdx"][lo:hi, :, :]
    dξdy_h = fid["dξdy"][lo:hi, :, :]
    dξdz_h = fid["dξdz"][lo:hi, :, :]
    dηdx_h = fid["dηdx"][lo:hi, :, :]
    dηdy_h = fid["dηdy"][lo:hi, :, :]
    dηdz_h = fid["dηdz"][lo:hi, :, :]
    dζdx_h = fid["dζdx"][lo:hi, :, :]
    dζdy_h = fid["dζdy"][lo:hi, :, :]
    dζdz_h = fid["dζdz"][lo:hi, :, :]

    J_h = fid["J"][lo:hi, :, :] 
    x_h = fid["x"][lo:hi, :, :] 
    y_h = fid["y"][lo:hi, :, :] 
    z_h = fid["z"][lo:hi, :, :]
    if IBM
        tag_h = fid["tag"][lo:hi, :, :]
        proj_h = fid["proj"][lo:hi, :, :, :]
        tag = CuArray(tag_h)
        proj = CuArray(proj_h)
    else
        tag_h = zeros(Int64, Nx_tot, Ny_tot, Nz_tot)
        tag = CuArray(tag_h)
    end
    close(fid)

    # move to device memory
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
    Yi =   CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
    ϕ  =   CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot) # Shock sensor
    U  =   CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fp =   CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fm =   CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fx =   CUDA.zeros(Float64, Nxp+1, Ny, Nz, Ncons)
    Fy =   CUDA.zeros(Float64, Nxp, Ny+1, Nz, Ncons)
    Fz =   CUDA.zeros(Float64, Nxp, Ny, Nz+1, Ncons)
    Fv_x = CUDA.zeros(Float64, Nxp+1, Ny, Nz, 4)
    Fv_y = CUDA.zeros(Float64, Nxp, Ny+1, Nz, 4)
    Fv_z = CUDA.zeros(Float64, Nxp, Ny, Nz+1, 4)

    Fp_i = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
    Fm_i = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
    Fx_i = CUDA.zeros(Float64, Nxp+1, Ny, Nz, Nspecs) # species advection
    Fy_i = CUDA.zeros(Float64, Nxp, Ny+1, Nz, Nspecs) # species advection
    Fz_i = CUDA.zeros(Float64, Nxp, Ny, Nz+1, Nspecs) # species advection
    Fd_x = CUDA.zeros(Float64, Nxp+1, Ny, Nz, Nspecs) # species diffusion
    Fd_y = CUDA.zeros(Float64, Nxp, Ny+1, Nz, Nspecs) # species diffusion
    Fd_z = CUDA.zeros(Float64, Nxp, Ny, Nz+1, Nspecs) # species diffusion
    Fhx = CUDA.zeros(Float64, Nxp+1, Ny, Nz, 3) # enthalpy diffusion
    Fhy = CUDA.zeros(Float64, Nxp, Ny+1, Nz, 3) # enthalpy diffusion
    Fhz = CUDA.zeros(Float64, Nxp, Ny, Nz+1, 3) # enthalpy diffusion

    μ = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    λ = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    D = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
    
    Un = similar(U)
    ρn = similar(ρi)

    # MPI buffer 
    Qsbuf_h = zeros(Float64, NG, Ny_tot, Nz_tot, Nprim)
    Qrbuf_h = similar(Qsbuf_h)
    dsbuf_h = zeros(Float64, NG, Ny_tot, Nz_tot, Nspecs)
    drbuf_h = similar(dsbuf_h)
    Mem.pin(Qsbuf_h)
    Mem.pin(Qrbuf_h)
    Mem.pin(dsbuf_h)
    Mem.pin(dsbuf_h)

    Qsbuf_d = CuArray(Qsbuf_h)
    Qrbuf_d = CuArray(Qrbuf_h)
    dsbuf_d = CuArray(dsbuf_h)
    drbuf_d = CuArray(drbuf_h)

    # initial
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock prim2c(U, Q)
    exchange_ghost(Q, Nprim, rank, comm, Qsbuf_h, Qsbuf_d, Qrbuf_h, Qrbuf_d)
    exchange_ghost(ρi, Nspecs, rank, comm, dsbuf_h, dsbuf_d, drbuf_h, drbuf_d)
    MPI.Barrier(comm)
    fillGhost(Q, U, ρi, Yi, thermo, rank)
    fillSpec(ρi)
    if IBM
        @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fillIB(Q, U, ρi, tag, proj, thermo)
    end

    if reaction
        if Luxmodel
            @load "./NN/Air/luxmodel.jld2" model ps st

            ps = ps |> gpu_device()

            w1 = ps[1].weight
            b1 = ps[1].bias
            w2 = ps[2].weight
            b2 = ps[2].bias
            w3 = ps[3].weight
            b3 = ps[3].bias

            Y1 = CUDA.ones(Float32, 64, Nxp*Ny*Nz)
            Y2 = CUDA.ones(Float32, 256, Nxp*Ny*Nz)
            yt_pred = CUDA.ones(Float32, Nspecs+1, Nxp*Ny*Nz)

            j = JSON.parsefile("./NN/Air/norm.json")
            lambda = j["lambda"]
            inputs_mean = CuArray(convert(Vector{Float32}, j["inputs_mean"]))
            inputs_std =  CuArray(convert(Vector{Float32}, j["inputs_std"]))
            labels_mean = CuArray(convert(Vector{Float32}, j["labels_mean"]))
            labels_std =  CuArray(convert(Vector{Float32}, j["labels_std"]))

            inputs = CUDA.zeros(Float32, Nspecs+2, Nxp*Ny*Nz)
            inputs_norm = CUDA.zeros(Float32, Nspecs+2, Nxp*Ny*Nz)
        end
        
        if Cantera
            # CPU evaluation needed
            inputs_h = zeros(Float64, Nspecs+2, Nxp*Ny*Nz)
            Mem.pin(inputs_h)
            inputs = CuArray(inputs_h)
        end

        dt2 = dt/2
    end

    for tt = 1:ceil(Int, Time/dt)
        if tt*dt > Time || tt > maxStep
            return
        end

        if reaction
            # Reaction Step
            if Luxmodel
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock getY(Yi, ρi, Q, tag)
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock pre_input(inputs, inputs_norm, Q, Yi, lambda, inputs_mean, inputs_std)
                evalModel(Y1, Y2, yt_pred, w1, w2, w3, b1, b2, b3, inputs_norm)
                @. yt_pred = yt_pred * labels_std + labels_mean
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock post_predict(yt_pred, inputs, U, Q, ρi, dt2, lambda, thermo)
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock c2Prim(U, Q, ρi, thermo, tag)
                exchange_ghost(Q, Nprim, rank, comm, Qsbuf_h, Qsbuf_d, Qrbuf_h, Qrbuf_d)
                exchange_ghost(ρi, Nspecs, rank, comm, dsbuf_h, dsbuf_d, drbuf_h, drbuf_d)
                MPI.Barrier(comm)
                fillGhost(Q, U, ρi, Yi, thermo, rank)
                fillSpec(ρi)
                if IBM
                    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fillIB(Q, U, ρi, tag, proj, thermo)
                end
            elseif Cantera
                # CPU - cantera
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock pre_input_cpu(inputs, Q, ρi)
                copyto!(inputs_h, inputs)
                eval_cpu(inputs_h, dt2)
                copyto!(inputs, inputs_h)
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock post_eval_cpu(inputs, U, Q, ρi, thermo)
                exchange_ghost(Q, Nprim, rank, comm, Qsbuf_h, Qsbuf_d, Qrbuf_h, Qrbuf_d)
                exchange_ghost(ρi, Nspecs, rank, comm, dsbuf_h, dsbuf_d, drbuf_h, drbuf_d)
                MPI.Barrier(comm)
                fillGhost(Q, U, ρi, Yi, thermo, rank)
                fillSpec(ρi)
                if IBM
                    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fillIB(Q, U, ρi, tag, proj, thermo)
                end
            else
                # GPU
                for _ = 1:sub_step
                    if stiff
                        @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock eval_gpu_stiff(U, Q, ρi, dt2/sub_step, thermo, react, tag)
                    else
                        @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock eval_gpu(U, Q, ρi, dt2/sub_step, thermo, react, tag)
                    end
                end
                exchange_ghost(Q, Nprim, rank, comm, Qsbuf_h, Qsbuf_d, Qrbuf_h, Qrbuf_d)
                exchange_ghost(ρi, Nspecs, rank, comm, dsbuf_h, dsbuf_d, drbuf_h, drbuf_d)
                MPI.Barrier(comm)
                fillGhost(Q, U, ρi, Yi, thermo, rank)
                fillSpec(ρi)
                if IBM
                    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fillIB(Q, U, ρi, tag, proj, thermo)
                end
            end
        end

        # RK3
        for KRK = 1:3
            if KRK == 1
                copyto!(Un, U)
                copyto!(ρn, ρi)
            end

            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock mixture(Q, ρi, Yi, λ, μ, D, thermo, tag)
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock shockSensor(ϕ, Q)
            specAdvance(ρi, Q, Yi, Fp_i, Fm_i, Fx_i, Fy_i, Fz_i, Fd_x, Fd_y, Fd_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, D, Fhx, Fhy, Fhz, thermo, consts, tag)
            flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, λ, μ, Fhx, Fhy, Fhz, consts, tag)

            if KRK == 2
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock linComb(U, Un, Ncons, 0.25, 0.75, tag)
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock linComb(ρi, ρn, Nspecs, 0.25, 0.75, tag)
            elseif KRK == 3
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock linComb(U, Un, Ncons, 2/3, 1/3, tag)
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock linComb(ρi, ρn, Nspecs, 2/3, 1/3, tag)
            end

            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock c2Prim(U, Q, ρi, thermo, tag)
            exchange_ghost(Q, Nprim, rank, comm, Qsbuf_h, Qsbuf_d, Qrbuf_h, Qrbuf_d)
            exchange_ghost(ρi, Nspecs, rank, comm, dsbuf_h, dsbuf_d, drbuf_h, drbuf_d)
            MPI.Barrier(comm)
            fillGhost(Q, U, ρi, Yi, thermo, rank)
            fillSpec(ρi)
            if IBM
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fillIB(Q, U, ρi, tag, proj, thermo)
            end
        end

        if reaction
            # Reaction Step
            if Luxmodel
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock getY(Yi, ρi, Q, tag)
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock pre_input(inputs, inputs_norm, Q, Yi, lambda, inputs_mean, inputs_std)
                evalModel(Y1, Y2, yt_pred, w1, w2, w3, b1, b2, b3, inputs_norm)
                @. yt_pred = yt_pred * labels_std + labels_mean
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock post_predict(yt_pred, inputs, U, Q, ρi, dt2, lambda, thermo)
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock c2Prim(U, Q, ρi, thermo, tag)
                exchange_ghost(Q, Nprim, rank, comm, Qsbuf_h, Qsbuf_d, Qrbuf_h, Qrbuf_d)
                exchange_ghost(ρi, Nspecs, rank, comm, dsbuf_h, dsbuf_d, drbuf_h, drbuf_d)
                MPI.Barrier(comm)
                fillGhost(Q, U, ρi, Yi, thermo, rank)
                fillSpec(ρi)
                if IBM
                    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fillIB(Q, U, ρi, tag, proj, thermo)
                end
            elseif Cantera
                # CPU - cantera
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock pre_input_cpu(inputs, Q, ρi)
                copyto!(inputs_h, inputs)
                eval_cpu(inputs_h, dt2)
                copyto!(inputs, inputs_h)
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock post_eval_cpu(inputs, U, Q, ρi, thermo)
                exchange_ghost(Q, Nprim, rank, comm, Qsbuf_h, Qsbuf_d, Qrbuf_h, Qrbuf_d)
                exchange_ghost(ρi, Nspecs, rank, comm, dsbuf_h, dsbuf_d, drbuf_h, drbuf_d)
                MPI.Barrier(comm)
                fillGhost(Q, U, ρi, Yi, thermo, rank)
                fillSpec(ρi)
                if IBM
                    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fillIB(Q, U, ρi, tag, proj, thermo)
                end
            else
                # GPU
                for _ = 1:sub_step
                    if stiff
                        @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock eval_gpu_stiff(U, Q, ρi, dt2/sub_step, thermo, react, tag)
                    else
                        @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock eval_gpu(U, Q, ρi, dt2/sub_step, thermo, react, tag)
                    end
                end
                exchange_ghost(Q, Nprim, rank, comm, Qsbuf_h, Qsbuf_d, Qrbuf_h, Qrbuf_d)
                exchange_ghost(ρi, Nspecs, rank, comm, dsbuf_h, dsbuf_d, drbuf_h, drbuf_d)
                MPI.Barrier(comm)
                fillGhost(Q, U, ρi, Yi, thermo, rank)
                fillSpec(ρi)
                if IBM
                    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fillIB(Q, U, ρi, tag, proj, thermo)
                end
            end
        end

        if tt % 10 == 0
            if rank == 0
                printstyled("Step: ", color=:cyan)
                print("$tt")
                printstyled("\tTime: ", color=:blue)
                println("$(tt*dt)")
                flush(stdout)
            end
            if any(isnan, U)
                printstyled("Oops, NaN detected\n", color=:red)
                return
            end
        end
        # Output
        if plt_out && (tt % step_plt == 0 || abs(Time-dt*tt) < dt || tt == maxStep)
            copyto!(Q_h, Q)
            copyto!(ρi_h, ρi)
            copyto!(ϕ_h, ϕ)

            # visualization file, in Float32
            mkpath("./PLT")
            fname::String = string("./PLT/plt", rank, "-", tt)

            rho = convert(Array{Float32, 3}, @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 1])
            vel = convert(Array{Float32, 4}, @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 2:4])
            p =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 5])
            T =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 6])
        
            YH2  = convert(Array{Float32, 3}, @view ρi_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 1])
            YO2  = convert(Array{Float32, 3}, @view ρi_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 2])
            YH2O = convert(Array{Float32, 3}, @view ρi_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 3])
            YH   = convert(Array{Float32, 3}, @view ρi_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 4])
            YOH  = convert(Array{Float32, 3}, @view ρi_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 6])
            YN2  = convert(Array{Float32, 3}, @view ρi_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 9])

            ϕ_ng = convert(Array{Float32, 3}, @view ϕ_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG])
            x_ng = convert(Array{Float32, 3}, @view x_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG])
            y_ng = convert(Array{Float32, 3}, @view y_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG])
            z_ng = convert(Array{Float32, 3}, @view z_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG])

            tag_ng = @view tag_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG]

            vtk_grid(fname, x_ng, y_ng, z_ng; compress=plt_compress_level) do vtk
                vtk["rho"] = rho
                vtk["velocity"] = @views (vel[:, :, :, 1], vel[:, :, :, 2], vel[:, :, :, 3])
                vtk["p"] = p
                vtk["T"] = T
                vtk["phi"] = ϕ_ng
                vtk["YH2"] = YH2
                vtk["YO2"] = YO2
                vtk["YH2O"] = YH2O
                vtk["YH"] = YH
                vtk["YOH"] = YOH
                vtk["YN2"] = YN2
                vtk["tag"] = tag_ng
                vtk["Time", VTKFieldData()] = dt * tt
            end 
        end

        # restart file, in Float64
        if chk_out && (tt % step_chk == 0 || abs(Time-dt*tt) < dt || tt == maxStep)
            copyto!(Q_h, Q)
            copyto!(ρi_h, ρi)

            mkpath("./CHK")
            chkname::String = string("./CHK/chk", tt, ".h5")
            h5open(chkname, "w", comm) do f
                dset1 = create_dataset(
                    f,
                    "Q_h",
                    datatype(Float64),
                    dataspace(Nx_tot, Ny_tot, Nz_tot, Nprim, Nprocs);
                    chunk=(Nx_tot, Ny_tot, Nz_tot, Nprim, 1),
                    compress=chk_compress_level,
                    dxpl_mpio=:collective
                )
                dset1[:, :, :, :, rank + 1] = Q_h
                dset2 = create_dataset(
                    f,
                    "ρi_h",
                    datatype(Float64),
                    dataspace(Nx_tot, Ny_tot, Nz_tot, Nspecs, Nprocs);
                    chunk=(Nx_tot, Ny_tot, Nz_tot, Nspecs, 1),
                    compress=chk_compress_level,
                    dxpl_mpio=:collective
                )
                dset2[:, :, :, :, rank + 1] = ρi_h
            end
        end
    end
    if rank == 0
        printstyled("Done!\n", color=:green)
        flush(stdout)
    end
    MPI.Barrier(comm)
    return
end
