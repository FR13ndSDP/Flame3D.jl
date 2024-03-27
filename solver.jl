using MPI
using WriteVTK
using LinearAlgebra, StaticArrays, CUDA
using CUDA:i32
using HDF5, DelimitedFiles

CUDA.allowscalar(false)

include("split.jl")
include("schemes.jl")
include("viscous.jl")
include("boundary.jl")
include("utils.jl")
include("div.jl")
include("mpi.jl")

function flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, μratio_x, μratio_y, μratio_z)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit(Q, Fp, Fm, dξdx, dξdy, dξdz)
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock WENO_x(Fx, ϕ, Fp, Fm, Ncons)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit(Q, Fp, Fm, dηdx, dηdy, dηdz)
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock WENO_y(Fy, ϕ, Fp, Fm, Ncons)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit(Q, Fp, Fm, dζdx, dζdy, dζdz)
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock WENO_z(Fz, ϕ, Fp, Fm, Ncons)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock viscousFlux_x(Fv_x, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, μratio_x)
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock viscousFlux_y(Fv_y, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, μratio_y)
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock viscousFlux_z(Fv_z, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, μratio_z)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock div(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J)
end

function time_step(rank, comm)
    Nx_tot = Nxp+2*NG
    Ny_tot = Nyp+2*NG
    Nz_tot = Nzp+2*NG

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
        close(fid)

        inlet_h = readdlm("./SCU-benchmark/flow-inlet.dat")

        Q  =   cu(Q_h)
        inlet  =   cu(inlet_h)
    else
        Q_h = zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Nprim)
        Q = CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Nprim)

        inlet_h = readdlm("./SCU-benchmark/flow-inlet.dat")

        copyto!(Q_h, Q)
        inlet  =   cu(inlet_h)

        initialize(Q, inlet)
    end
    
    ϕ_h = zeros(Float32, Nx_tot, Ny_tot, Nz_tot) # shock sensor

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
    close(fid)

    # move to device memory
    dξdx = cu(dξdx_h)
    dξdy = cu(dξdy_h)
    dξdz = cu(dξdz_h)
    dηdx = cu(dηdx_h)
    dηdy = cu(dηdy_h)
    dηdz = cu(dηdz_h)
    dζdx = cu(dζdx_h)
    dζdy = cu(dζdy_h)
    dζdz = cu(dζdz_h)
    J = cu(J_h)

    # allocate on device
    ϕ  =   CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot) # Shock sensor
    U  =   CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fp =   CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fm =   CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fx =   CUDA.zeros(Float32, Nxp+1, Nyp, Nzp, Ncons)
    Fy =   CUDA.zeros(Float32, Nxp, Nyp+1, Nzp, Ncons)
    Fz =   CUDA.zeros(Float32, Nxp, Nyp, Nzp+1, Ncons)
    Fv_x = CUDA.zeros(Float32, Nxp+1, Nyp, Nzp, 4)
    Fv_y = CUDA.zeros(Float32, Nxp, Nyp+1, Nzp, 4)
    Fv_z = CUDA.zeros(Float32, Nxp, Nyp, Nzp+1, 4)
    μratio_x = CUDA.zeros(Float32, Nxp+1, Nyp, Nzp)
    μratio_y = CUDA.zeros(Float32, Nxp, Nyp+1, Nzp)
    μratio_z = CUDA.zeros(Float32, Nxp, Nyp, Nzp+1)

    Un = similar(U)

    if average
        Q_avg = CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Nprim)
    end

    # MPI buffer 
    Qsbuf_h = zeros(Float32, NG, Ny_tot, Nz_tot, Nprim)
    Qrbuf_h = similar(Qsbuf_h)
    Mem.pin(Qsbuf_h)
    Mem.pin(Qrbuf_h)

    Qsbuf_d = cu(Qsbuf_h)
    Qrbuf_d = cu(Qrbuf_h)

    # # initial
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock prim2c(U, Q)
    exchange_ghost(Q, Nprim, rank, comm, Qsbuf_h, Qsbuf_d, Qrbuf_h, Qrbuf_d)
    MPI.Barrier(comm)
    fillGhost(Q, U, rank, inlet)

    for tt = 1:ceil(Int, Time/dt)
        if tt*dt > Time || tt > maxStep
            return
        end

        # RK3
        for KRK = 1:3
            if KRK == 1
                copyto!(Un, U)
            end

            # @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock localstep(Q, dt_local, 0.1, x, y, z)
            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock shockSensor(ϕ, Q)
            flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, μratio_x, μratio_y, μratio_z)

            if KRK == 2
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock linComb(U, Un, Ncons, 0.25f0, 0.75f0)
            elseif KRK == 3
                @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock linComb(U, Un, Ncons, 2/3f0, 1/3f0)
            end

            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock c2Prim(U, Q)
            exchange_ghost(Q, Nprim, rank, comm, Qsbuf_h, Qsbuf_d, Qrbuf_h, Qrbuf_d)
            MPI.Barrier(comm)
            fillGhost(Q, U, rank, inlet)
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

            μratio_hx = zeros(Float32, Nxp+1, Nyp, Nzp)
            μratio_hy = zeros(Float32, Nxp, Nyp+1, Nzp)
            μratio_hz = zeros(Float32, Nxp, Nyp, Nzp+1)
            copyto!(μratio_hx, μratio_x)
            copyto!(μratio_hy, μratio_y)
            copyto!(μratio_hz, μratio_z)

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

            μration_ngx = @view μratio_hx[1:Nxp, :, :]
            μration_ngy = @view μratio_hy[:, 1:Nyp, :]
            μration_ngz = @view μratio_hz[:, :, 1:Nzp]

            vtk_grid(fname, x_ng, y_ng, z_ng; compress=plt_compress_level) do vtk
                vtk["rho"] = rho
                vtk["u"] = u
                vtk["v"] = v
                vtk["w"] = w
                vtk["p"] = p
                vtk["T"] = T
                vtk["ratiox"] = μration_ngx
                vtk["ratioy"] = μration_ngy
                vtk["ratioz"] = μration_ngz
                vtk["phi"] = ϕ_ng
            end 
            μratio_hx = nothing
            μratio_hy = nothing
            μratio_hz = nothing
        end

        # restart file, in Float32
        if chk_out && (tt % step_chk == 0 || abs(Time-dt*tt) < dt || tt == maxStep)
            copyto!(Q_h, Q)

            mkpath("./CHK")
            chkname::String = string("./CHK/chk", tt, ".h5")
            h5open(chkname, "w", comm) do f
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
                x_ng = @view x_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG]
                y_ng = @view y_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG]
                z_ng = @view z_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG]
                
                rho = @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 1]
                u   = @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 2]
                v   = @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 3]
                w   = @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 4]
                p =   @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 5]
                T =   @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 6]

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
    MPI.Barrier(comm)
    return
end
