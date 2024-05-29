function fill_x(Q, U, ρi, Yi, thermo, rank)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
        return
    end
    # inlet
    if rank == 0 && i <= NG+1
        for n = 1:Nspecs
            @inbounds Yi[i, j, k, n] = 0.f0
            @inbounds ρi[i, j, k, n] = 0.f0
        end
        if (j-68)^2+(k-68)^2 < 225
            @inbounds Yi[i, j, k, 1] = 0.15f0
            @inbounds Yi[i, j, k, 9] = 0.85f0
            P = 101325.f0
            T = 305.f0
            @inbounds Y = @view Yi[i, j, k, :]
            rho = ρmixture(P, T, Y, thermo)
            @inbounds ρi[i, j, k, 9] = Yi[i, j, k, 9] * rho
            @inbounds ρi[i, j, k, 1] = Yi[i, j, k, 1] * rho
            @inbounds rhoi = @view ρi[i, j, k, :]
            ei = InternalEnergy(T, rhoi, thermo)
            u = 900.f0 #sqrt(γ*P/rho) * Ma
        else
            @inbounds Yi[i, j, k, 9] = 0.767f0
            @inbounds Yi[i, j, k, 2] = 0.233f0
            P = 101325.f0
            T = 1150.f0
            @inbounds Y = @view Yi[i, j, k, :]
            rho = ρmixture(P, T, Y, thermo)
            @inbounds ρi[i, j, k, 9] = Yi[i, j, k, 9] * rho
            @inbounds ρi[i, j, k, 2] = Yi[i, j, k, 2] * rho
            @inbounds rhoi = @view ρi[i, j, k, :]
            ei = InternalEnergy(T, rhoi, thermo)
            u = 20.f0
        end
        @inbounds Q[i, j, k, 1] = rho
        @inbounds Q[i, j, k, 2] = u
        @inbounds Q[i, j, k, 3] = 0.f0
        @inbounds Q[i, j, k, 4] = 0.f0
        @inbounds Q[i, j, k, 5] = P
        @inbounds Q[i, j, k, 6] = T
        @inbounds Q[i, j, k, 7] = ei

        @inbounds U[i, j, k, 1] = rho
        @inbounds U[i, j, k, 2] = u * rho
        @inbounds U[i, j, k, 3] = 0.f0
        @inbounds U[i, j, k, 4] = 0.f0
        @inbounds U[i, j, k, 5] = ei + 0.5f0*rho*u^2
    end

    if rank == Nprocs[1]-1 && i >= Nxp+NG
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
    
    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
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
    
    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
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
    
    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
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
    
    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
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
function fillGhost(Q, U, ρi, Yi, thermo, rank)
    @cuda threads=nthreads blocks=nblock fill_x(Q, U, ρi, Yi, thermo, rank)
    @cuda threads=nthreads blocks=nblock fill_y(Q, U)
    @cuda threads=nthreads blocks=nblock fill_z(Q, U)
end

# only in two trival directions
function fillSpec(ρi)
    @cuda threads=nthreads blocks=nblock fill_y_s(ρi)
    @cuda threads=nthreads blocks=nblock fill_z_s(ρi)
end

function init(Q, ρi, ρ, u, v, w, P, T, T_ignite, ρ_ig, thermo)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    for n = 1:Nspecs
        @inbounds ρi[i, j, k, n] = 0.f0
    end

    # ignite area
    if (j-68)^2+(k-68)^2 < 225
        rho = ρ_ig
        temp = T_ignite
        @inbounds ρi[i, j, k, 1] = rho * 0.15f0
        @inbounds ρi[i, j, k, 9] = rho * 0.85f0
        uu = 900.f0
    else
        rho = ρ
        temp = T
        @inbounds ρi[i, j, k, 2] = rho * 0.233f0
        @inbounds ρi[i, j, k, 9] = rho * 0.767f0
        uu = u
    end

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
    T::Float32 = 1150.f0
    T_ignite::Float32 = 305.f0
    P::Float32 = 101325.f0
    gas.TPY = T, P, "O2:0.233 N2:0.767"
    ρ::Float32 = gas.density
    gas.TPY = T_ignite, P, "H2:0.15 N2:0.85"
    ρ_ig::Float32 = gas.density
    u::Float32 = 20.f0
    v::Float32 = 0.f0
    w::Float32 = 0.f0
    
    @cuda threads=nthreads blocks=nblock init(Q, ρi, ρ, u, v, w, P, T, T_ignite, ρ_ig, thermo)
end