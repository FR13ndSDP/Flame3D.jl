using MPI
using WriteVTK
using LinearAlgebra, StaticArrays, AMDGPU
using HDF5, DelimitedFiles

AMDGPU.allowscalar(false)

include("split.jl")
include("schemes.jl")
include("viscous.jl")
include("boundary.jl")
include("utils.jl")
include("div.jl")
include("mpi.jl")

function flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ)

    @roc groupsize=nthreads gridsize=ngroups fluxSplit(Q, Fp, Fm, dξdx, dξdy, dξdz)
    @roc groupsize=nthreads gridsize=ngroups WENO_x(Fx, ϕ, Fp, Fm, Ncons)
    @roc groupsize=nthreads gridsize=ngroups viscousFlux_x(Fx, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J)

    @roc groupsize=nthreads gridsize=ngroups fluxSplit(Q, Fp, Fm, dηdx, dηdy, dηdz)
    @roc groupsize=nthreads gridsize=ngroups WENO_y(Fy, ϕ, Fp, Fm, Ncons)
    @roc groupsize=nthreads gridsize=ngroups viscousFlux_y(Fy, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J)

    @roc groupsize=nthreads gridsize=ngroups fluxSplit(Q, Fp, Fm, dζdx, dζdy, dζdz)
    @roc groupsize=nthreads gridsize=ngroups WENO_z(Fz, ϕ, Fp, Fm, Ncons)
    @roc groupsize=nthreads gridsize=ngroups viscousFlux_z(Fz, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J)

    @roc groupsize=nthreads gridsize=ngroups div(U, Fx, Fy, Fz, dt, J)
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

    if restart[end-2:end] == ".h5"
        if rank == 0
            printstyled("Restart\n", color=:yellow)
        end
        fid = h5open(restart, "r", comm_cart)
        Q_h = fid["Q_h"][lox:hix, loy:hiy, loz:hiz, :, rank+1]
        close(fid)

        inlet_h = readdlm("./SCU-benchmark/flow-inlet.dat")

        Q  =   ROCArray(Q_h)
        inlet  =   ROCArray(inlet_h)
    else
        Q_h = zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Nprim)
        Q = AMDGPU.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Nprim)

        inlet_h = readdlm("./SCU-benchmark/flow-inlet.dat")

        copyto!(Q_h, Q)
        inlet  =   ROCArray(inlet_h)

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
    dξdx = ROCArray(dξdx_h)
    dξdy = ROCArray(dξdy_h)
    dξdz = ROCArray(dξdz_h)
    dηdx = ROCArray(dηdx_h)
    dηdy = ROCArray(dηdy_h)
    dηdz = ROCArray(dηdz_h)
    dζdx = ROCArray(dζdx_h)
    dζdy = ROCArray(dζdy_h)
    dζdz = ROCArray(dζdz_h)
    J = ROCArray(J_h)

    # allocate on device
    ϕ  =   AMDGPU.zeros(Float32, Nx_tot, Ny_tot, Nz_tot) # Shock sensor
    U  =   AMDGPU.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fp =   AMDGPU.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fm =   AMDGPU.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fx =   AMDGPU.zeros(Float32, Nxp+1, Nyp, Nzp, Ncons)
    Fy =   AMDGPU.zeros(Float32, Nxp, Nyp+1, Nzp, Ncons)
    Fz =   AMDGPU.zeros(Float32, Nxp, Nyp, Nzp+1, Ncons)

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

    Qsbuf_dx = ROCArray(Qsbuf_hx)
    Qsbuf_dy = ROCArray(Qsbuf_hy)
    Qsbuf_dz = ROCArray(Qsbuf_hz)
    Qrbuf_dx = ROCArray(Qrbuf_hx)
    Qrbuf_dy = ROCArray(Qrbuf_hy)
    Qrbuf_dz = ROCArray(Qrbuf_hz)

    # # initial
    @roc groupsize=nthreads gridsize=ngroups prim2c(U, Q)
    exchange_ghost(Q, Nprim, comm_cart, 
                   Qsbuf_hx, Qsbuf_dx, Qrbuf_hx, Qrbuf_dx,
                   Qsbuf_hy, Qsbuf_dy, Qrbuf_hy, Qrbuf_dy,
                   Qsbuf_hz, Qsbuf_dz, Qrbuf_hz, Qrbuf_dz)
    MPI.Barrier(comm_cart)
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
            flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ)

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
            MPI.Barrier(comm_cart)
            fillGhost(Q, U, rankx, ranky, inlet)
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
            copyto!(ϕ_h, ϕ)

            # visualization file, in Float32
            mkpath("./PLT")
            fname::String = string("./PLT/plt", rank, "-", tt)

            rho = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 1]
            u   = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 2]
            v   = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 3]
            w   = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 4]
            p   = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 5]
            T   = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 6]

            ϕ_ng = @view ϕ_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG]
            x_ng = @view x_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG]
            y_ng = @view y_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG]
            z_ng = @view z_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG]

            vtk_grid(fname, x_ng, y_ng, z_ng; compress=plt_compress_level) do vtk
                vtk["rho"] = rho
                vtk["u"] = u
                vtk["v"] = v
                vtk["w"] = w
                vtk["p"] = p
                vtk["T"] = T
                vtk["phi"] = ϕ_ng
            end 
        end

        # restart file, in Float32
        if chk_out && (tt % step_chk == 0 || abs(Time-dt*tt) < dt || tt == maxStep)
            copyto!(Q_h, Q)

            mkpath("./CHK")
            chkname::String = string("./CHK/chk", tt, ".h5")
            h5open(chkname, "w", comm_cart) do f
                dset1 = create_dataset(
                    f,
                    "Q_h",
                    datatype(Float32),
                    dataspace(Nx_tot, Ny_tot, Nz_tot, Nprim, Nprocs[1]*Nprocs[2]*Nprocs[3]);
                    chunk=(Nx_tot, Ny_tot, Nz_tot, Nprim, 1),
                    compress=chk_compress_level,
                    dxpl_mpio=:collective
                )
                dset1[:, :, :, :, rank + 1] = Q_h
            end
        end

        # Average output
        if average
            if tt % avg_step == 0
                @. Q_avg += Q/avg_total
            end

            if tt == avg_step*avg_total
                if rank == 0
                    printstyled("average done\n", color=:green)
                end

                mkpath("./PLT")
                avgname::String = string("./PLT/avg", rank, "-", tt)

                copyto!(Q_h, Q_avg)
                x_ng = @view x_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG]
                y_ng = @view y_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG]
                z_ng = @view z_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG]
                
                rho = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 1]
                u   = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 2]
                v   = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 3]
                w   = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 4]
                p =   @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 5]
                T =   @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 6]

                vtk_grid(avgname, x_ng, y_ng, z_ng; compress=plt_compress_level) do vtk
                    vtk["rho"] = rho
                    vtk["u"] = u
                    vtk["v"] = v
                    vtk["w"] = w
                    vtk["p"] = p
                    vtk["T"] = T
                    vtk["Time", VTKFieldData()] = dt * tt
                end 
                
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
