function plotFile_xdmf(tt, Q, ϕ, Q_h, ϕ_h, coords_h, comm_cart, rank, rankx, ranky, rankz)
    # Output
    if plt_out && (tt % step_plt == 0 || abs(Time-dt*tt) < dt || tt == maxStep)
        copyto!(Q_h, Q)
        copyto!(ϕ_h, ϕ)

        if rank == 0
            mkpath("./PLT")
            write_XDMF(tt)
        end

        ρ = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 1]
        u = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 2]
        v = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 3]
        w = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 4]
        p = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 5]
        T = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, 6]

        ϕ_ng = @view ϕ_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG]

        coords_ng = @view coords_h[:, 1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG]

        # global indices no ghost
        lox = rankx*Nxp+1
        hix = (rankx+1)*Nxp

        loy = ranky*Nyp+1
        hiy = (ranky+1)*Nyp

        loz = rankz*Nzp+1
        hiz = (rankz+1)*Nzp

        fname::String = string("./PLT/plt-", tt, ".h5")
        h5open(fname, "w", comm_cart) do f
            dset1 = create_dataset(
                f,
                "rho",
                datatype(Float32),
                dataspace(Nx, Ny, Nz);
                chunk=(Nxp, Nyp, Nzp),
                shuffle=plt_shuffle,
                compress=plt_compress_level,
                dxpl_mpio=:collective
            )
            dset1[lox:hix, loy:hiy, loz:hiz] = ρ
            dset2 = create_dataset(
                f,
                "u",
                datatype(Float32),
                dataspace(Nx, Ny, Nz);
                chunk=(Nxp, Nyp, Nzp),
                shuffle=plt_shuffle,
                compress=plt_compress_level,
                dxpl_mpio=:collective
            )
            dset2[lox:hix, loy:hiy, loz:hiz] = u
            dset3 = create_dataset(
                f,
                "v",
                datatype(Float32),
                dataspace(Nx, Ny, Nz);
                chunk=(Nxp, Nyp, Nzp),
                shuffle=plt_shuffle,
                compress=plt_compress_level,
                dxpl_mpio=:collective
            )
            dset3[lox:hix, loy:hiy, loz:hiz] = v
            dset4 = create_dataset(
                f,
                "w",
                datatype(Float32),
                dataspace(Nx, Ny, Nz);
                chunk=(Nxp, Nyp, Nzp),
                shuffle=plt_shuffle,
                compress=plt_compress_level,
                dxpl_mpio=:collective
            )
            dset4[lox:hix, loy:hiy, loz:hiz] = w
            dset5 = create_dataset(
                f,
                "p",
                datatype(Float32),
                dataspace(Nx, Ny, Nz);
                chunk=(Nxp, Nyp, Nzp),
                shuffle=plt_shuffle,
                compress=plt_compress_level,
                dxpl_mpio=:collective
            )
            dset5[lox:hix, loy:hiy, loz:hiz] = p
            dset6 = create_dataset(
                f,
                "T",
                datatype(Float32),
                dataspace(Nx, Ny, Nz);
                chunk=(Nxp, Nyp, Nzp),
                shuffle=plt_shuffle,
                compress=plt_compress_level,
                dxpl_mpio=:collective
            )
            dset6[lox:hix, loy:hiy, loz:hiz] = T
            dset7 = create_dataset(
                f,
                "phi",
                datatype(Float32),
                dataspace(Nx, Ny, Nz);
                chunk=(Nxp, Nyp, Nzp),
                shuffle=plt_shuffle,
                compress=plt_compress_level,
                dxpl_mpio=:collective
            )
            dset7[lox:hix, loy:hiy, loz:hiz] = ϕ_ng
            dset8 = create_dataset(
                f,
                "coords",
                datatype(Float32),
                dataspace(3, Nx, Ny, Nz);
                chunk=(3, Nxp, Nyp, Nzp),
                shuffle=plt_shuffle,
                compress=plt_compress_level,
                dxpl_mpio=:collective
            )
            dset8[:, lox:hix, loy:hiy, loz:hiz] = coords_ng
        end
    end
end

function plotFile_h5(tt, Q, ϕ, Q_h, ϕ_h, comm_cart, rank, rankx, ranky, rankz)
    # Output
    if plt_out && (tt % step_plt == 0 || abs(Time-dt*tt) < dt || tt == maxStep)
        copyto!(Q_h, Q)
        copyto!(ϕ_h, ϕ)

        if rank == 0
            mkpath("./PLT")
        end

        primitives = @view Q_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG, :]

        ϕ_ng = @view ϕ_h[1+NG:Nxp+NG, 1+NG:Nyp+NG, 1+NG:Nzp+NG]

        # global indices no ghost
        lox = rankx*Nxp+1
        hix = (rankx+1)*Nxp

        loy = ranky*Nyp+1
        hiy = (ranky+1)*Nyp

        loz = rankz*Nzp+1
        hiz = (rankz+1)*Nzp

        fname::String = string("./PLT/plt-", tt, ".h5")
        h5open(fname, "w", comm_cart) do f
            dset1 = create_dataset(
                f,
                "Q",
                datatype(Float32),
                dataspace(Nx, Ny, Nz, Nprim);
                chunk=(Nxp, Nyp, Nzp, Nprim),
                shuffle=plt_shuffle,
                compress=plt_compress_level,
                dxpl_mpio=:collective
            )
            dset1[lox:hix, loy:hiy, loz:hiz, :] = primitives
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

    fname::String = string("./AVG/avg-", tt, ".h5")
    h5open(fname, "w", comm_cart) do f
        dset1 = create_dataset(
            f,
            "avg",
            datatype(Float32),
            dataspace(Nx, Ny, Nz, Nprim);
            chunk=(Nxp, Nyp, Nzp, Nprim),
            shuffle=avg_shuffle,
            compress=avg_compress_level,
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

# XDMF metadata, note that julia in column major
function write_XDMF(tt)
    fname = string("./PLT/plt-", tt, ".xmf")
    h5name = string("plt-", tt, ".h5")
    time = tt*dt

    open(fname, "w") do f
        write(f, "<?xml version=\"1.0\" ?>")
        write(f, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n")
        write(f, "<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">\n")
        write(f, " <Domain>\n")
        write(f, "  <Grid Name=\"Grid\" GridType=\"Uniform\">\n")
        write(f, "  <Time Value=\"$time\" />\n")
        write(f, "   <Topology TopologyType=\"3DSMesh\" NumberOfElements=\"$Nz $Ny $Nx\" />\n")
        write(f, "   <Geometry GeometryType=\"XYZ\">\n")
        write(f, "   <DataItem Name=\"coords\" Dimensions=\"$Nz $Ny $Nx 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n")
        write(f, "    $h5name:/coords\n")
        write(f, "   </DataItem>\n")
        write(f, "   </Geometry>\n")
        write(f, "   <Attribute Name=\"rho\" AttributeType=\"Scalar\" Center=\"Node\">\n")
        write(f, "    <DataItem Dimensions=\"$Nz $Ny $Nx\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n")
        write(f, "    $h5name:/rho\n")
        write(f, "   </DataItem>\n")
        write(f, "   </Attribute>\n")
        write(f, "   <Attribute Name=\"u\" AttributeType=\"Scalar\" Center=\"Node\">\n")
        write(f, "    <DataItem Dimensions=\"$Nz $Ny $Nx\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n")
        write(f, "    $h5name:/u\n")
        write(f, "   </DataItem>\n")
        write(f, "   </Attribute>\n")
        write(f, "   <Attribute Name=\"v\" AttributeType=\"Scalar\" Center=\"Node\">\n")
        write(f, "    <DataItem Dimensions=\"$Nz $Ny $Nx\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n")
        write(f, "    $h5name:/v\n")
        write(f, "   </DataItem>\n")
        write(f, "   </Attribute>\n")
        write(f, "   <Attribute Name=\"w\" AttributeType=\"Scalar\" Center=\"Node\">\n")
        write(f, "    <DataItem Dimensions=\"$Nz $Ny $Nx\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n")
        write(f, "    $h5name:/w\n")
        write(f, "   </DataItem>\n")
        write(f, "   </Attribute>\n")
        write(f, "   <Attribute Name=\"p\" AttributeType=\"Scalar\" Center=\"Node\">\n")
        write(f, "    <DataItem Dimensions=\"$Nz $Ny $Nx\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n")
        write(f, "    $h5name:/p\n")
        write(f, "   </DataItem>\n")
        write(f, "   </Attribute>\n")
        write(f, "   <Attribute Name=\"T\" AttributeType=\"Scalar\" Center=\"Node\">\n")
        write(f, "    <DataItem Dimensions=\"$Nz $Ny $Nx\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n")
        write(f, "    $h5name:/T\n")
        write(f, "   </DataItem>\n")
        write(f, "   </Attribute>\n")
        write(f, "   <Attribute Name=\"phi\" AttributeType=\"Scalar\" Center=\"Node\">\n")
        write(f, "    <DataItem Dimensions=\"$Nz $Ny $Nx\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n")
        write(f, "    $h5name:/phi\n")
        write(f, "   </DataItem>\n")
        write(f, "   </Attribute>\n")
        write(f, "  </Grid>\n")
        write(f, " </Domain>\n")
        write(f, "</Xdmf>\n")
    end
end