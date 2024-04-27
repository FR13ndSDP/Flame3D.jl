function plotFile(tt,  Q, ϕ, Q_h, ϕ_h, x_h, y_h, z_h, rank, rankx, ranky, rankz, plt_files, extents)
    # Output
    if plt_out && (tt % step_plt == 0 || abs(Time-dt*tt) < dt || tt == maxStep)
        copyto!(Q_h, Q)
        copyto!(ϕ_h, ϕ)

        fname::String = string("./plt-", tt)
        
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

function checkpointFile(tt, Q_h, Q, comm_cart, rank)
    # restart file, in Float32
    if chk_out && (tt % step_chk == 0 || abs(Time-dt*tt) < dt || tt == maxStep)
        Nx_tot = Nxp+2*NG
        Ny_tot = Nyp+2*NG
        Nz_tot = Nzp+2*NG

        copyto!(Q_h, Q)

        if rank == 0
            mkpath("./CHK")
        end
        chkname::String = string("./CHK/chk-", tt, ".h5")
        h5open(chkname, "w", comm_cart) do f
            dset1 = create_dataset(
                f,
                "Q_h",
                datatype(Float32),
                dataspace(Nx_tot, Ny_tot, Nz_tot, Nprim, prod(Nprocs));
                chunk=(Nx_tot, Ny_tot, Nz_tot, Nprim, 1),
                shuffle=chk_shuffle,
                compress=chk_compress_level,
                dxpl_mpio=:collective
            )
            dset1[:, :, :, :, rank + 1] = Q_h
        end
    end
end

function averageFile(tt, Q_avg, Q_h, comm_cart, rankx, ranky, rankz)
    copyto!(Q_h, Q_avg)

    avg = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, :]

    # global indices no ghost
    lox = rankx*Nxp+1
    hix = (rankx+1)*Nxp

    loy = ranky*Nyp+1
    hiy = (ranky+1)*Nyp

    loz = rankz*Nzp+1
    hiz = (rankz+1)*Nzp

    chkname::String = string("./AVG/avg-", tt, ".h5")
    h5open(chkname, "w", comm_cart) do f
        dset1 = create_dataset(
            f,
            "avg",
            datatype(Float32),
            dataspace(Nx, Ny, Nz, Nprim);
            chunk=(Nxp, Nyp, Nzp, Nprim),
            shuffle=chk_shuffle,
            compress=chk_compress_level,
            dxpl_mpio=:collective
        )
        dset1[lox:hix, loy:hiy, loz:hiz, :] = avg
    end
end

# provide debug output with ghost cells
function debugOutput(Q, Q_h, x_h, y_h, z_h, rank)
    copyto!(Q_h, Q)

    fname::String = string("debug", "-", rank)

    rho = @view Q_h[:, :, :, 1]
    u   = @view Q_h[:, :, :, 2]
    v   = @view Q_h[:, :, :, 3]
    w   = @view Q_h[:, :, :, 4]
    p   = @view Q_h[:, :, :, 5]
    T   = @view Q_h[:, :, :, 6]

    vtk_grid(fname, x_h, y_h, z_h) do vtk
        vtk["rho"] = rho
        vtk["u"] = u
        vtk["v"] = v
        vtk["w"] = w
        vtk["p"] = p
        vtk["T"] = T
    end 
end