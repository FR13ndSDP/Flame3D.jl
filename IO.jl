function plotFile(tt, Q_h, ϕ_h, x_h, y_h, z_h, Q, ϕ, rank, rankx, ranky, rankz, plt_files, extents)
    # Output
    if plt_out && (tt % step_plt == 0 || abs(Time-dt*tt) < dt || tt == maxStep)
        copyto!(Q_h, Q)
        copyto!(ϕ_h, ϕ)

        # visualization file, in Float32
        # mkpath("./PLT")
        fname::String = string("./plt", "-", tt)
        
        xindex = ifelse(rankx == Nprocs[1]-1, 1+NG:Nxp+NG, 1+NG:Nxp+NG+1)
        yindex = ifelse(ranky == Nprocs[2]-1, 1+NG:Nyp+NG, 1+NG:Nyp+NG+1)
        zindex = ifelse(rankz == Nprocs[3]-1, 1+NG:Nzp+NG, 1+NG:Nzp+NG+1)

        rho = @view Q_h[xindex, yindex, zindex, 1]
        u   = @view Q_h[xindex, yindex, zindex, 2]
        v   = @view Q_h[xindex, yindex, zindex, 3]
        w   = @view Q_h[xindex, yindex, zindex, 4]
        p   = @view Q_h[xindex, yindex, zindex, 5]
        T   = @view Q_h[xindex, yindex, zindex, 6]

        ϕ_ng = @view ϕ_h[xindex, yindex, zindex]
        x_ng = @view x_h[xindex, yindex, zindex]
        y_ng = @view y_h[xindex, yindex, zindex]
        z_ng = @view z_h[xindex, yindex, zindex]

        plt_files[rank+1] = pvtk_grid(fname, x_ng, y_ng, z_ng; part=rank+1, extents=extents, compress=plt_compress_level) do pvtk
            pvtk["rho"] = rho
            pvtk["u"] = u
            pvtk["v"] = v
            pvtk["w"] = w
            pvtk["p"] = p
            pvtk["T"] = T
            pvtk["phi"] = ϕ_ng
            pvtk["Time", VTKFieldData()] = dt * tt
        end 
    end
end

function checkpointFile(tt, Q_h, Q, comm_cart)
    # restart file, in Float32
    if chk_out && (tt % step_chk == 0 || abs(Time-dt*tt) < dt || tt == maxStep)
        Nx_tot = Nxp+2*NG
        Ny_tot = Nyp+2*NG
        Nz_tot = Nzp+2*NG

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
end

function averageFile(tt, Q_avg, Q_h, x_h, y_h, z_h, rank, rankx, ranky, rankz, plt_files, extents)
    # mkpath("./PLT")
    avgname::String = string("./avg", "-", tt)

    copyto!(Q_h, Q_avg)

    xindex = ifelse(rankx == Nprocs[1]-1, 1+NG:Nxp+NG, 1+NG:Nxp+NG+1)
    yindex = ifelse(ranky == Nprocs[2]-1, 1+NG:Nyp+NG, 1+NG:Nyp+NG+1)
    zindex = ifelse(rankz == Nprocs[3]-1, 1+NG:Nzp+NG, 1+NG:Nzp+NG+1)

    x_ng = @view x_h[xindex, yindex, zindex]
    y_ng = @view y_h[xindex, yindex, zindex]
    z_ng = @view z_h[xindex, yindex, zindex]
    
    rho = @view Q_h[xindex, yindex, zindex, 1]
    u   = @view Q_h[xindex, yindex, zindex, 2]
    v   = @view Q_h[xindex, yindex, zindex, 3]
    w   = @view Q_h[xindex, yindex, zindex, 4]
    p =   @view Q_h[xindex, yindex, zindex, 5]
    T =   @view Q_h[xindex, yindex, zindex, 6]

    plt_files[rank+1] = pvtk_grid(avgname, x_ng, y_ng, z_ng; part=rank+1, extents=extents, compress=plt_compress_level) do pvtk
        pvtk["rho"] = rho
        pvtk["u"] = u
        pvtk["v"] = v
        pvtk["w"] = w
        pvtk["p"] = p
        pvtk["T"] = T
        pvtk["Time", VTKFieldData()] = dt * tt
    end 
end