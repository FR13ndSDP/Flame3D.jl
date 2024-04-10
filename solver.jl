using MPI
using WriteVTK
using StaticArrays, AMDGPU
using HDF5, DelimitedFiles
using Dates, Printf

include("split.jl")
include("schemes.jl")
include("viscous.jl")
include("boundary.jl")
include("utils.jl")
include("div.jl")
include("mpi.jl")
include("IO.jl")

function flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, s1, s2, s3, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ)

    @roc groupsize=nthreads gridsize=ngroups fluxSplit(Q, Fp, Fm, s1, dξdx, dξdy, dξdz)
    @roc groupsize=nthreads gridsize=ngroups WENO_x(Fx, ϕ, s1, Fp, Fm, Ncons)

    @roc groupsize=nthreads gridsize=ngroups fluxSplit(Q, Fp, Fm, s2, dηdx, dηdy, dηdz)
    @roc groupsize=nthreads gridsize=ngroups WENO_y(Fy, ϕ, s2, Fp, Fm, Ncons)

    @roc groupsize=nthreads gridsize=ngroups fluxSplit(Q, Fp, Fm, s3, dζdx, dζdy, dζdz)
    @roc groupsize=nthreads gridsize=ngroups WENO_z(Fz, ϕ, s3, Fp, Fm, Ncons)

    @roc groupsize=nthreads gridsize=ngroups viscousFlux(Fv_x, Fv_y, Fv_z, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J)

    @roc groupsize=nthreads gridsize=ngroups div(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J)
end

function time_step(rank, comm_cart)
    Nx_tot = Nxp+2*NG
    Ny_tot = Nyp+2*NG
    Nz_tot = Nzp+2*NG

    # global indices
    (rankx, ranky, rankz) = MPI.Cart_coords(comm_cart, rank)

    lox = rankx*Nxp+1
    hix = (rankx+1)*Nxp+2*NG

    loy = ranky*Nyp+1
    hiy = (ranky+1)*Nyp+2*NG

    loz = rankz*Nzp+1
    hiz = (rankz+1)*Nzp+2*NG

    # prepare pvtk metadata, kind of ugly
    total_ranks = Nprocs[1]*Nprocs[2]*Nprocs[3]
    plt_files = Vector{Vector{String}}(undef, total_ranks)  # files saved by each process
    extents = Vector{Tuple{UnitRange{Int64}, UnitRange{Int64}, UnitRange{Int64}}}(undef, total_ranks)
    for n = 1:total_ranks
        (is, js, ks) = MPI.Cart_coords(comm_cart, n-1)

        lx = is*Nxp+1
        hx = min((is+1)*Nxp+1, Nx)
    
        ly = js*Nyp+1
        hy = min((js+1)*Nyp+1, Ny)
    
        lz = ks*Nzp+1
        hz = min((ks+1)*Nzp+Nzp+1, Nz)

        extents[n] = (lx:hx, ly:hy, lz:hz)
    end

    if restart[end-2:end] == ".h5"
        if rank == 0
            printstyled("Restart\n", color=:yellow)
        end
        fid = h5open(restart, "r", comm_cart)
        Q_h = fid["Q_h"][:, :, :, :, rank+1]
        close(fid)

        inlet_h = readdlm("./SCU-benchmark/flow-inlet.dat", Float32)

        Q = ROCArray(Q_h)
        inlet = ROCArray(inlet_h)
    else
        Q_h = zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Nprim)
        Q = AMDGPU.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Nprim)

        inlet_h = readdlm("./SCU-benchmark/flow-inlet.dat", Float32)

        copyto!(Q_h, Q)
        inlet = ROCArray(inlet_h)

        initialize(Q, inlet, ranky)
    end
    
    ϕ_h = zeros(Float32, Nx_tot, Ny_tot, Nz_tot) # shock sensor

    # load mesh metrics
    fid = h5open("metrics.h5", "r", comm_cart)
    dξdx_h = fid["dξdx"][lox:hix, loy:hiy, loz:hiz]
    dξdy_h = fid["dξdy"][lox:hix, loy:hiy, loz:hiz]
    dξdz_h = fid["dξdz"][lox:hix, loy:hiy, loz:hiz]
    dηdx_h = fid["dηdx"][lox:hix, loy:hiy, loz:hiz]
    dηdy_h = fid["dηdy"][lox:hix, loy:hiy, loz:hiz]
    dηdz_h = fid["dηdz"][lox:hix, loy:hiy, loz:hiz]
    dζdx_h = fid["dζdx"][lox:hix, loy:hiy, loz:hiz]
    dζdy_h = fid["dζdy"][lox:hix, loy:hiy, loz:hiz]
    dζdz_h = fid["dζdz"][lox:hix, loy:hiy, loz:hiz]

    J_h = fid["J"][lox:hix, loy:hiy, loz:hiz] 
    x_h = fid["x"][lox:hix, loy:hiy, loz:hiz] 
    y_h = fid["y"][lox:hix, loy:hiy, loz:hiz] 
    z_h = fid["z"][lox:hix, loy:hiy, loz:hiz]
    close(fid)

    # move to device memory
    dξdx = ROCArray(convert(Array{Float32, 3}, dξdx_h))
    dξdy = ROCArray(convert(Array{Float32, 3}, dξdy_h))
    dξdz = ROCArray(convert(Array{Float32, 3}, dξdz_h))
    dηdx = ROCArray(convert(Array{Float32, 3}, dηdx_h))
    dηdy = ROCArray(convert(Array{Float32, 3}, dηdy_h))
    dηdz = ROCArray(convert(Array{Float32, 3}, dηdz_h))
    dζdx = ROCArray(convert(Array{Float32, 3}, dζdx_h))
    dζdy = ROCArray(convert(Array{Float32, 3}, dζdy_h))
    dζdz = ROCArray(convert(Array{Float32, 3}, dζdz_h))
    J = ROCArray(convert(Array{Float32, 3}, J_h))
    s1 = @. sqrt(dξdx^2+dξdy^2+dξdz^2)
    s2 = @. sqrt(dηdx^2+dηdy^2+dηdz^2)
    s3 = @. sqrt(dζdx^2+dζdy^2+dζdz^2)

    # allocate on device
    ϕ  =   AMDGPU.zeros(Float32, Nx_tot, Ny_tot, Nz_tot) # Shock sensor
    U  =   AMDGPU.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fp =   AMDGPU.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fm =   AMDGPU.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fx =   AMDGPU.zeros(Float32, Nxp+1, Nyp, Nzp, Ncons)
    Fy =   AMDGPU.zeros(Float32, Nxp, Nyp+1, Nzp, Ncons)
    Fz =   AMDGPU.zeros(Float32, Nxp, Nyp, Nzp+1, Ncons)
    Fv_x = AMDGPU.zeros(Float32, Nxp+NG, Nyp+NG, Nzp+NG, 4)
    Fv_y = AMDGPU.zeros(Float32, Nxp+NG, Nyp+NG, Nzp+NG, 4)
    Fv_z = AMDGPU.zeros(Float32, Nxp+NG, Nyp+NG, Nzp+NG, 4)

    Un = similar(U)

    if average
        Q_avg = AMDGPU.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Nprim)
    end

    # MPI buffer 
    Qsbuf_hx = zeros(Float32, NG, Ny_tot, Nz_tot, Nprim)
    Qsbuf_hy = zeros(Float32, Nx_tot, NG, Nz_tot, Nprim)
    Qsbuf_hz = zeros(Float32, Nx_tot, Ny_tot, NG, Nprim)
    Qrbuf_hx = similar(Qsbuf_hx)
    Qrbuf_hy = similar(Qsbuf_hy)
    Qrbuf_hz = similar(Qsbuf_hz)

    Qsbuf_dx = unsafe_wrap(ROCArray, pointer(Qsbuf_hx), size(Qsbuf_hx))
    Qsbuf_dy = unsafe_wrap(ROCArray, pointer(Qsbuf_hy), size(Qsbuf_hy))
    Qsbuf_dz = unsafe_wrap(ROCArray, pointer(Qsbuf_hz), size(Qsbuf_hz))
    Qrbuf_dx = unsafe_wrap(ROCArray, pointer(Qrbuf_hx), size(Qrbuf_hx))
    Qrbuf_dy = unsafe_wrap(ROCArray, pointer(Qrbuf_hy), size(Qrbuf_hy))
    Qrbuf_dz = unsafe_wrap(ROCArray, pointer(Qrbuf_hz), size(Qrbuf_hz))

    # initial
    @roc groupsize=nthreads gridsize=ngroups prim2c(U, Q)
    exchange_ghost(Q, Nprim, comm_cart, 
                   Qsbuf_hx, Qsbuf_dx, Qrbuf_hx, Qrbuf_dx,
                   Qsbuf_hy, Qsbuf_dy, Qrbuf_hy, Qrbuf_dy,
                   Qsbuf_hz, Qsbuf_dz, Qrbuf_hz, Qrbuf_dz)
    fillGhost(Q, U, rankx, ranky, inlet)

    for tt = 1:ceil(Int, Time/dt)
        if tt*dt > Time || tt > maxStep
            return
        end

        # RK3
        for KRK = 1:3
            if KRK == 1
                copyto!(Un, U)
            end

            @roc groupsize=nthreads gridsize=ngroups shockSensor(ϕ, Q)
            flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz,  Fv_x, Fv_y, Fv_z, s1, s2, s3, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ)

            if KRK == 2
                @roc groupsize=nthreads gridsize=ngroups linComb(U, Un, Ncons, 0.25f0, 0.75f0)
            elseif KRK == 3
                @roc groupsize=nthreads gridsize=ngroups linComb(U, Un, Ncons, 2/3f0, 1/3f0)
            end

            @roc groupsize=nthreads gridsize=ngroups c2Prim(U, Q)
            exchange_ghost(Q, Nprim, comm_cart, 
                           Qsbuf_hx, Qsbuf_dx, Qrbuf_hx, Qrbuf_dx,
                           Qsbuf_hy, Qsbuf_dy, Qrbuf_hy, Qrbuf_dy,
                           Qsbuf_hz, Qsbuf_dz, Qrbuf_hz, Qrbuf_dz)
            fillGhost(Q, U, rankx, ranky, inlet)
        end

        if tt % 100 == 0 && rank == 0
            printstyled("Step: ", color=:cyan)
            @printf "%g" tt
            printstyled("\tTime: ", color=:blue)
            @printf "%.2e" tt*dt
            printstyled("\tWall time: ", color=:green)
            println("$(now())")
            flush(stdout)

            if any(isnan, U)
                printstyled("Oops, NaN detected\n", color=:red)
                flush(stdout)
                MPI.Abort(comm_cart, 1)
                return
            end
        end

        plotFile(tt, Q, ϕ, Q_h, ϕ_h, x_h, y_h, z_h, rank, rankx, ranky, rankz, plt_files, extents)

        checkpointFile(tt, Q_h, Q, comm_cart)

        # Average output
        if average
            if tt % avg_step == 0
                @. Q_avg += Q/avg_total
            end

            if tt == avg_step*avg_total
                if rank == 0
                    printstyled("average done\n", color=:green)
                end

                averageFile(tt, Q_avg, Q_h, x_h, y_h, z_h, rank, rankx, ranky, rankz, plt_files, extents)
                
                return
            end
        end
    end
    if rank == 0
        printstyled("Done!\n", color=:green)
        flush(stdout)
    end
    MPI.Barrier(comm_cart)
    return
end
