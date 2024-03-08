function fill_x(Q, U, ρi, Yi, thermo, rank)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end
    # inlet
    if rank == 0 && i <= NG+1
        for n = 1:Nspecs
            @inbounds Yi[i, j, k, n] = 0.0
            @inbounds ρi[i, j, k, n] = 0.0
        end
        # if (j-36)^2+(k-36)^2 < 225
        #     @inbounds Yi[i, j, k, 9] = 0.767
        #     @inbounds Yi[i, j, k, 2] = 0.233
        #     P = 101325.0
        #     T = 1150.0
        #     @inbounds Y = @view Yi[i, j, k, :]
        #     rho = ρmixture(P, T, Y, thermo)
        #     @inbounds ρi[i, j, k, 9] = Yi[i, j, k, 9] * rho
        #     @inbounds ρi[i, j, k, 2] = Yi[i, j, k, 2] * rho
        #     @inbounds rhoi = @view ρi[i, j, k, :]
        #     ei = InternalEnergy(T, rhoi, thermo)
        #     u = 20.0
        # else
        @inbounds Yi[i, j, k, 9] = 0.767
        @inbounds Yi[i, j, k, 2] = 0.233
        P = 101325.0
        T = 290.0
        @inbounds Y = @view Yi[i, j, k, :]
        rho = ρmixture(P, T, Y, thermo)
        @inbounds ρi[i, j, k, 9] = Yi[i, j, k, 9] * rho
        @inbounds ρi[i, j, k, 2] = Yi[i, j, k, 2] * rho
        @inbounds rhoi = @view ρi[i, j, k, :]
        ei = InternalEnergy(T, rhoi, thermo)
        u = 280.0
        # end
        @inbounds Q[i, j, k, 1] = rho
        @inbounds Q[i, j, k, 2] = u
        @inbounds Q[i, j, k, 3] = 0
        @inbounds Q[i, j, k, 4] = 0
        @inbounds Q[i, j, k, 5] = P
        @inbounds Q[i, j, k, 6] = T
        @inbounds Q[i, j, k, 7] = ei

        @inbounds U[i, j, k, 1] = rho
        @inbounds U[i, j, k, 2] = u * rho
        @inbounds U[i, j, k, 3] = 0
        @inbounds U[i, j, k, 4] = 0
        @inbounds U[i, j, k, 5] = ei + 0.5*rho*u^2
    end

    if rank == Nprocs-1 && i >= Nxp+NG
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[Nxp+NG-1, j, k, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[Nxp+NG-1, j, k, n]
        end
        for n = 1:Nspecs
            @inbounds ρi[i, j, k, n] = ρi[Nxp+NG-1, j, k, n]
        end
    end
    return
end

function fill_y(Q, U)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    if j <= NG+1
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, NG+2, k, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, NG+2, k, n]
        end
    elseif j > Ny+NG-1
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, Ny+NG-1, k, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, Ny+NG-1, k, n]
        end
    end
    return
end

function fill_z(Q, U)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    if k <= NG+1
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, j, NG+2, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, j, NG+2, n]
        end
    elseif k > Nz+NG-1
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, j, Nz+NG-1, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, j, Nz+NG-1, n]
        end
    end
    return
end

function fill_y_s(ρi)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    if j <= NG+1
        for n = 1:Nspecs
            @inbounds ρi[i, j, k, n] = ρi[i, NG+2, k, n]
        end
    elseif j > Ny+NG-1
        for n = 1:Nspecs
            @inbounds ρi[i, j, k, n] = ρi[i, Ny+NG-1, k, n]
        end
    end
    return
end

function fill_z_s(ρi)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    # periodic
    if k <= NG+1
        for n = 1:Nspecs
            @inbounds ρi[i, j, k, n] = ρi[i, j, NG+2, n]
        end
    elseif k >= Nz+NG
        for n = 1:Nspecs
            @inbounds ρi[i, j, k, n] = ρi[i, j, Nz+NG-1, n]
        end
    end
    return
end

# special treatment on wall
# fill Q and U
function fillGhost(Q, U, ρi, Yi, thermo, rank)
    @cuda threads=nthreads blocks=nblock fill_x(Q, U, ρi, Yi, thermo, rank)
    @cuda threads=nthreads blocks=nblock fill_y(Q, U)
    @cuda threads=nthreads blocks=nblock fill_z(Q, U)
end

# only in two trival directions
# fill ρ only
function fillSpec(ρi)
    @cuda threads=nthreads blocks=nblock fill_y_s(ρi)
    @cuda threads=nthreads blocks=nblock fill_z_s(ρi)
end

function fillIB(Q, U, ρi, tag, proj, thermo)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    Tw::Float64 = 290.0

    if tag[i, j, k] == 2
        ii = proj[i, j, k, 1]
        jj = proj[i, j, k, 2]
        kk = proj[i, j, k, 3]

        Y_IB = MVector{Nspecs, Float64}(undef)
        ρinv = 1/Q[i+ii, j+jj, k+kk, 1]
        for n = 1:Nspecs
            @inbounds Y_IB[n] = ρi[i+ii, j+jj, k+kk, n] * ρinv
        end

        Pw =  Q[i+ii, j+jj, k+kk, 5]
        Q[i ,j ,k ,2] = 0
        Q[i ,j ,k ,3] = 0
        Q[i ,j ,k ,4] = 0
        Q[i ,j ,k ,5] = Pw
        Q[i ,j ,k ,6] = Tw
        Q[i ,j ,k ,1] = ρmixture(Pw, Tw, Y_IB, thermo)

        ρ_IB = @view ρi[i, j, k, :]
        for n = 1:Nspecs
            @inbounds ρ_IB[n] = Y_IB[n] * Q[i, j, k, 1]
        end
        Q[i, j, k, 7] = InternalEnergy(Tw, ρ_IB, thermo)

        U[i, j, k, 1] = Q[i, j, k, 1]
        U[i, j, k, 2] = 0
        U[i, j, k, 3] = 0
        U[i, j, k, 4] = 0
        U[i, j, k, 5] = Q[i, j, k, 7]
    elseif tag[i, j, k] == 3
        ii = proj[i, j, k, 1]
        jj = proj[i, j, k, 2]
        kk = proj[i, j, k, 3]

        Y_IB = MVector{Nspecs, Float64}(undef)
        ρinv = 1/Q[i+ii, j+jj, k+kk, 1]
        for n = 1:Nspecs
            @inbounds Y_IB[n] = ρi[i+ii, j+jj, k+kk, n] * ρinv
        end

        p_proj = Q[i+ii, j+jj, k+kk, 5]
        u_proj = Q[i+ii, j+jj, k+kk, 2]
        v_proj = Q[i+ii, j+jj, k+kk, 3]
        w_proj = Q[i+ii, j+jj, k+kk, 4]
        T_proj = Q[i+ii, j+jj, k+kk, 6]

        Q[i ,j ,k ,2] = -u_proj
        Q[i ,j ,k ,3] = -v_proj
        Q[i ,j ,k ,4] = -w_proj
        Q[i ,j ,k ,5] = p_proj
        TIB = 2*Tw-T_proj
        Q[i, j, k, 6] = TIB
        ρ = ρmixture(p_proj, TIB, Y_IB, thermo)
        Q[i, j, k, 1] = ρ

        ρ_IB = @view ρi[i, j, k, :]
        for n = 1:Nspecs
            @inbounds ρ_IB[n] = Y_IB[n] * ρ
        end
        Q[i, j, k, 7] = InternalEnergy(TIB, ρ_IB, thermo)

        U[i, j, k, 1] = ρ
        U[i, j, k, 2] = -ρ * u_proj
        U[i, j, k, 3] = -ρ * v_proj
        U[i, j, k, 4] = -ρ * w_proj
        U[i, j, k, 5] = Q[i, j, k, 7] + 0.5*ρ*(u_proj^2+v_proj^2+w_proj^2)
    end
    return
end

function init(Q, ρi, ρ, u, v, w, P, T, T_ignite, ρ_ig, thermo)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    for n = 1:Nspecs
        @inbounds ρi[i, j, k, n] = 0.0
    end

    # ignite area
    # if (j-68)^2+(k-68)^2 < 225
    #     rho = ρ_ig
    #     temp = T_ignite
    #     @inbounds ρi[i, j, k, 1] = rho * 0.15
    #     @inbounds ρi[i, j, k, 9] = rho * 0.85
    #     uu = 900
    # else
    rho = ρ
    temp = T
    @inbounds ρi[i, j, k, 2] = rho * 0.233
    @inbounds ρi[i, j, k, 9] = rho * 0.767
    uu = u
    # end

    # fill H2
    @inbounds rhoi = @view ρi[i, j, k, :]

    @inbounds Q[i, j, k, 1] = rho
    @inbounds Q[i, j, k, 2] = uu
    @inbounds Q[i, j, k, 3] = v
    @inbounds Q[i, j, k, 4] = w
    @inbounds Q[i, j, k, 5] = P
    @inbounds Q[i, j, k, 6] = temp
    @inbounds Q[i, j, k, 7] = InternalEnergy(temp, rhoi, thermo)
    return
end

#initialization on GPU
function initialize(Q, ρi, thermo)
    ct = pyimport("cantera")
    gas = ct.Solution(mech)
    T::Float64 = 290.0
    T_ignite::Float64 = 305.0
    P::Float64 = 101325.0
    gas.TPY = T, P, "O2:0.233 N2:0.767"
    ρ::Float64 = gas.density
    gas.TPY = T_ignite, P, "H2:0.15 N2:0.85"
    ρ_ig::Float64 = gas.density
    u::Float64 = 280.0
    v::Float64 = 0.0
    w::Float64 = 0.0
    
    @cuda threads=nthreads blocks=nblock init(Q, ρi, ρ, u, v, w, P, T, T_ignite, ρ_ig, thermo)
end