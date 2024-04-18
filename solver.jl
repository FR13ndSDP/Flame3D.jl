using MPI
using WriteVTK
using StaticArrays, CUDA
using CUDA:i32
using HDF5, DelimitedFiles
using Dates, Printf

CUDA.allowscalar(false)

include("split.jl")
include("schemes.jl")
include("viscous.jl")
include("boundary.jl")
include("utils.jl")
include("div.jl")
include("mpi.jl")
include("IO.jl")

function flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, s1, s2, s3, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ)

    if splitMethod == "SW"
        @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_SW(Q, Fp, Fm, s1, dξdx, dξdy, dξdz)
    else
        @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_LF(Q, Fp, Fm, s1, dξdx, dξdy, dξdz)
    end
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock advect_x(Fx, ϕ, s1, Fp, Fm, Ncons)

    if splitMethod == "SW"
        @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_SW(Q, Fp, Fm, s2, dηdx, dηdy, dηdz)
    else
        @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_LF(Q, Fp, Fm, s2, dηdx, dηdy, dηdz)
    end
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock advect_y(Fy, ϕ, s2, Fp, Fm, Ncons)

    if splitMethod == "SW"
        @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_SW(Q, Fp, Fm, s3, dζdx, dζdy, dζdz)
    else
        @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock fluxSplit_LF(Q, Fp, Fm, s3, dζdx, dζdy, dζdz)
    end
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock advect_z(Fz, ϕ, s3, Fp, Fm, Ncons)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock viscousFlux(Fv_x, Fv_y, Fv_z, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J)

    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock div(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J)
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
    total_ranks = prod(Nprocs)
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
        Q_h = fid["Q_h"][lox:hix, loy:hiy, loz:hiz, :, 1]
        close(fid)

        inlet_h = readdlm("./SCU-benchmark/flow-inlet.dat", Float32)

        Q = cu(Q_h)
        inlet = cu(inlet_h)
    else
        Q_h = zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Nprim)
        Q = CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Nprim)

        inlet_h = readdlm("./SCU-benchmark/flow-inlet.dat", Float32)

        copyto!(Q_h, Q)
        inlet = cu(inlet_h)

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
    s1 = @. sqrt(dξdx^2+dξdy^2+dξdz^2)
    s2 = @. sqrt(dηdx^2+dηdy^2+dηdz^2)
    s3 = @. sqrt(dζdx^2+dζdy^2+dζdz^2)

    # allocate on device
    ϕ  =   CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot) # Shock sensor
    U  =   CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fp =   CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fm =   CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fx =   CUDA.zeros(Float32, Nxp+1, Nyp, Nzp, Ncons)
    Fy =   CUDA.zeros(Float32, Nxp, Nyp+1, Nzp, Ncons)
    Fz =   CUDA.zeros(Float32, Nxp, Nyp, Nzp+1, Ncons)
    Fv_x = CUDA.zeros(Float32, Nxp+NG, Nyp+NG, Nzp+NG, 4)
    Fv_y = CUDA.zeros(Float32, Nxp+NG, Nyp+NG, Nzp+NG, 4)
    Fv_z = CUDA.zeros(Float32, Nxp+NG, Nyp+NG, Nzp+NG, 4)

    Un = similar(U)

    if average
        Q_avg = CUDA.zeros(Float32, Nx_tot, Ny_tot, Nz_tot, Nprim)
    end

    # MPI buffer 
    Qsbuf_hx = zeros(Float32, NG, Ny_tot, Nz_tot, Nprim)
    Qsbuf_hy = zeros(Float32, Nx_tot, NG, Nz_tot, Nprim)
    Qsbuf_hz = zeros(Float32, Nx_tot, Ny_tot, NG, Nprim)
    Qrbuf_hx = similar(Qsbuf_hx)
    Qrbuf_hy = similar(Qsbuf_hy)
    Qrbuf_hz = similar(Qsbuf_hz)
    Mem.pin(Qsbuf_hx)
    Mem.pin(Qsbuf_hy)
    Mem.pin(Qsbuf_hz)
    Mem.pin(Qrbuf_hx)
    Mem.pin(Qrbuf_hy)
    Mem.pin(Qrbuf_hz)

    Qsbuf_dx = cu(Qsbuf_hx)
    Qsbuf_dy = cu(Qsbuf_hy)
    Qsbuf_dz = cu(Qsbuf_hz)
    Qrbuf_dx = cu(Qrbuf_hx)
    Qrbuf_dy = cu(Qrbuf_hy)
    Qrbuf_dz = cu(Qrbuf_hz)

    # # initial
    @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock prim2c(U, Q)
    exchange_ghost(Q, Nprim, comm_cart, 
                   Qsbuf_hx, Qsbuf_dx, Qrbuf_hx, Qrbuf_dx,
                   Qsbuf_hy, Qsbuf_dy, Qrbuf_hy, Qrbuf_dy,
                   Qsbuf_hz, Qsbuf_dz, Qrbuf_hz, Qrbuf_dz)
    fillGhost(Q, U, rankx, ranky, inlet)

    # sampling metadata
    if sample
        sample_count::Int64 = 1
        valid_rankx = -1
        valid_ranky = -1
        valid_rankz = -1
    
        # find target ranks
        if sample_index[1] ≠ -1
            local_rankx::Int64 = sample_index[1] ÷ Nxp
            local_idx::Int64 = sample_index[1] % Nxp
    
            if rankx == local_rankx
                valid_rankx = rank
            end
    
            # collect on rank 0
            if rank == 0
                collectionx = zeros(Ny, Nz, Nprim, sample_total)
                rank_listx = MPI.gather(valid_rankx, comm_cart)
                rank_listx = filter!(x->x!=-1, rank_listx)
            else
                MPI.gather(valid_rankx, comm_cart)
            end
        end
    
        if sample_index[2] ≠ -1
            local_ranky::Int64 = sample_index[2] ÷ Nyp
            local_idy::Int64 = sample_index[2] % Nyp
    
            if ranky == local_ranky
                valid_ranky = rank
            end
    
            # collect on rank 0
            if rank == 0
                collectiony = zeros(Nx, Nz, Nprim, sample_total)
                rank_listy = MPI.gather(valid_ranky, comm_cart)
                rank_listy = filter!(x->x!=-1, rank_listy)
            else
                MPI.gather(valid_ranky, comm_cart)
            end
        end
    
        if sample_index[3] ≠ -1
            local_rankz::Int64 = sample_index[3] ÷ Nzp
            local_idz::Int64 = sample_index[3] % Nzp
    
            if rankz == local_rankz
                valid_rankz = rank
            end
    
            # collect on rank 0
            if rank == 0
                collectionz = zeros(Nx, Ny, Nprim, sample_total)
                rank_listz = MPI.gather(valid_rankz, comm_cart)
                rank_listz = filter!(x->x!=-1, rank_listz)
            else
                MPI.gather(valid_rankz, comm_cart)
            end
        end
    end

    for tt = 1:ceil(Int, Time/dt)
        if tt*dt > Time || tt > maxStep
            return
        end

        # RK3
        for KRK = 1:3
            if KRK == 1
                copyto!(Un, U)
            end

            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock shockSensor(ϕ, Q)
            flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, s1, s2, s3, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ)

            if KRK == 2
                @cuda maxregs=maxreg fastmath=true threads=nthreads2 blocks=nblock2 linComb(U, Un, Ncons, 0.25f0, 0.75f0)
            elseif KRK == 3
                @cuda maxregs=maxreg fastmath=true threads=nthreads2 blocks=nblock2 linComb(U, Un, Ncons, 2/3f0, 1/3f0)
            end

            @cuda maxregs=maxreg fastmath=true threads=nthreads blocks=nblock c2Prim(U, Q)
            exchange_ghost(Q, Nprim, comm_cart, 
                           Qsbuf_hx, Qsbuf_dx, Qrbuf_hx, Qrbuf_dx,
                           Qsbuf_hy, Qsbuf_dy, Qrbuf_hy, Qrbuf_dy,
                           Qsbuf_hz, Qsbuf_dz, Qrbuf_hz, Qrbuf_dz)
            fillGhost(Q, U, rankx, ranky, inlet)
        end

        if tt % 10 == 0 && rank == 0
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

        checkpointFile(tt, Q_h, Q, comm_cart, rank)

        # Average output
        if average && tt <= avg_step*avg_total
            if tt % avg_step == 0
                @. Q_avg += Q/avg_total
            end

            if tt == avg_step*avg_total
                if rank == 0
                    printstyled("average done\n", color=:green)
                end

                averageFile(tt, Q_avg, Q_h, x_h, y_h, z_h, rank, rankx, ranky, rankz, plt_files, extents)                
            end
        end

        # collection of slice
        if sample && (tt % sample_step == 0) && sample_count <= sample_total

            if sample_index[1] ≠ -1
                if rankx == local_rankx && rank ≠ 0
                    copyto!(Q_h, Q)
                    part = @view Q_h[local_idx, 1+NG:Nyp+NG, 1+NG:Nzp+NG, :]
                    MPI.Send(part, 0, 0, comm_cart)
                end
            
                if rank == 0
                    copyto!(Q_h, Q)
                    part = @view Q_h[local_idx, 1+NG:Nyp+NG, 1+NG:Nzp+NG, :]
                    for i ∈ rank_listx
                        if i ≠ 0
                            MPI.Recv!(part, i, 0, comm_cart)
                        end
            
                        # get global index
                        (_, ry, rz) = MPI.Cart_coords(comm_cart, i)
            
                        ly = ry*Nyp+1
                        hy = (ry+1)*Nyp
            
                        lz = rz*Nzp+1
                        hz = (rz+1)*Nzp
            
                        collectionx[ly:hy, lz:hz, :, sample_count] = part
                    end
                end
            end

            if sample_index[2] ≠ -1
                if ranky == local_ranky && rank ≠ 0
                    copyto!(Q_h, Q)
                    part = @view Q_h[1+NG:Nxp+NG, local_idy, 1+NG:Nzp+NG, :]
                    MPI.Send(part, 0, 0, comm_cart)
                end
            
                if rank == 0
                    copyto!(Q_h, Q)
                    part = @view Q_h[1+NG:Nxp+NG, local_idy, 1+NG:Nzp+NG, :]
                    for i ∈ rank_listy
                        if i ≠ 0
                            MPI.Recv!(part, i, 0, comm_cart)
                        end

                        # get global index
                        (rx, _, rz) = MPI.Cart_coords(comm_cart, i)
            
                        lx = rx*Nxp+1
                        hx = (rx+1)*Nxp
            
                        lz = rz*Nzp+1
                        hz = (rz+1)*Nzp
            
                        collectiony[lx:hx, lz:hz, :, sample_count] = part
                    end
                end
            end
            
            if sample_index[3] ≠ -1
                if rankz == local_rankz && rank ≠ 0
                    copyto!(Q_h, Q)
                    part = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, local_idz, :]
                    MPI.Send(part, 0, 0, comm_cart)
                end
            
                if rank == 0
                    copyto!(Q_h, Q)
                    part = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, local_idz, :]
                    for i ∈ rank_listz
                        if i ≠ 0
                            MPI.Recv!(part, i, 0, comm_cart)
                        end

                        # get global index
                        (rx, ry, _) = MPI.Cart_coords(comm_cart, i)
            
                        lx = rx*Nxp+1
                        hx = (rx+1)*Nxp
            
                        ly = ry*Nyp+1
                        hy = (ry+1)*Nyp
            
                        collectionz[lx:hx, ly:hy, :, sample_count] = part
                    end
                end
            end

            if sample_count == sample_total && rank == 0
                mkpath("./SAMPLE")

                if sample_index[1] ≠ -1
                    h5open("./SAMPLE/collection-x.h5", "w") do file
                        file["collection"] = collectionx
                    end
                end

                if sample_index[2] ≠ -1
                    h5open("./SAMPLE/collection-y.h5", "w") do file
                        file["collection"] = collectiony
                    end
                end

                if sample_index[3] ≠ -1
                    h5open("./SAMPLE/collection-z.h5", "w") do file
                        file["collection"] = collectionz
                    end
                end
            end
            sample_count += 1
        end
    end
    if rank == 0
        printstyled("Done!\n", color=:green)
        flush(stdout)
    end
    MPI.Barrier(comm_cart)
    return
end