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

        @inbounds Yi[i, j, k, 5] = 0.767
        @inbounds Yi[i, j, k, 2] = 0.233
        P = 79.8
        T = 270.65 
        @inbounds Y = @view Yi[i, j, k, :]
        rho = ρmixture(P, T, Y, thermo)
        @inbounds ρi[i, j, k, 5] = Yi[i, j, k, 5] * rho
        @inbounds ρi[i, j, k, 2] = Yi[i, j, k, 2] * rho
        @inbounds rhoi = @view ρi[i, j, k, :]
        ei = InternalEnergy(T, rhoi, thermo)
        u = 1000.0
        w = 0.0
        v = 0.0

        @inbounds Q[i, j, k, 1] = rho
        @inbounds Q[i, j, k, 2] = u
        @inbounds Q[i, j, k, 3] = v
        @inbounds Q[i, j, k, 4] = w
        @inbounds Q[i, j, k, 5] = P
        @inbounds Q[i, j, k, 6] = T
        @inbounds Q[i, j, k, 7] = ei

        @inbounds U[i, j, k, 1] = rho
        @inbounds U[i, j, k, 2] = u * rho
        @inbounds U[i, j, k, 3] = v * rho
        @inbounds U[i, j, k, 4] = w * rho
        @inbounds U[i, j, k, 5] = ei + 0.5*rho*(u^2+v^2+w^2)
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

function fill_z(Q, U, ρi, Yi, thermo)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    if k <= NG+1
        for n = 1:Nspecs
            @inbounds Yi[i, j, k, n] = 0.0
            @inbounds ρi[i, j, k, n] = 0.0
        end

        @inbounds Yi[i, j, k, 5] = 0.767
        @inbounds Yi[i, j, k, 2] = 0.233
        P = 79.8
        T = 270.65 
        @inbounds Y = @view Yi[i, j, k, :]
        rho = ρmixture(P, T, Y, thermo)
        @inbounds ρi[i, j, k, 5] = Yi[i, j, k, 5] * rho
        @inbounds ρi[i, j, k, 2] = Yi[i, j, k, 2] * rho
        @inbounds rhoi = @view ρi[i, j, k, :]
        ei = InternalEnergy(T, rhoi, thermo)
        u = 1000.0
        w = 0.0
        v = 0.0

        @inbounds Q[i, j, k, 1] = rho
        @inbounds Q[i, j, k, 2] = u
        @inbounds Q[i, j, k, 3] = v
        @inbounds Q[i, j, k, 4] = w
        @inbounds Q[i, j, k, 5] = P
        @inbounds Q[i, j, k, 6] = T
        @inbounds Q[i, j, k, 7] = ei

        @inbounds U[i, j, k, 1] = rho
        @inbounds U[i, j, k, 2] = u * rho
        @inbounds U[i, j, k, 3] = v * rho
        @inbounds U[i, j, k, 4] = w * rho
        @inbounds U[i, j, k, 5] = ei + 0.5*rho*(u^2+v^2+w^2)
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
    if k >= Nz+NG
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
    @cuda threads=nthreads blocks=nblock fill_z(Q, U, ρi, Yi, thermo)
end

# only in two trival directions
# fill ρ only
function fillSpec(ρi)
    @cuda threads=nthreads blocks=nblock fill_y_s(ρi)
    @cuda threads=nthreads blocks=nblock fill_z_s(ρi)
end

function fillIB(Q, U, ρi, neari, nearj, neark, c_d, c_n, tag, thermo)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    Tw::Float64 = 300.0
    np = 15

    if tag[i, j, k] == 3
        interpi = @view neari[:, i, j, k]
        interpj = @view nearj[:, i, j, k]
        interpk = @view neark[:, i, j, k]

        # YIB: zero gradient
        Yg = MVector{Nspecs, Float64}(undef)
        for n = 1:Nspecs
            sum = 0.0
            for nn = 1:np
                Ynear = ρi[interpi[nn], interpj[nn], interpk[nn], n]/Q[interpi[nn], interpj[nn], interpk[nn], 1]
                @inbounds sum += c_n[nn+1, i, j, k] * Ynear
            end
            @inbounds Yg[n] = -sum/c_n[1, i, j, k]
        end

        # P： zero gradient
        sum = 0.0
        for nn = 1:np
            @inbounds sum += c_n[nn+1, i, j, k] * Q[interpi[nn], interpj[nn], interpk[nn], 5]
        end
        @inbounds pg = -sum/c_n[1, i, j, k]
        
        # u,v,w,T: dirichlet
        sum = 0.0
        for nn = 1:np
            @inbounds sum += c_d[nn+1, i, j, k] * Q[interpi[nn], interpj[nn], interpk[nn], 2]
        end
        @inbounds ug = -sum/c_d[1, i, j, k]

        sum = 0.0
        for nn = 1:np
            @inbounds sum += c_d[nn+1, i, j, k] * Q[interpi[nn], interpj[nn], interpk[nn], 3]
        end
        @inbounds vg = -sum/c_d[1, i, j, k]

        sum = 0.0
        for nn = 1:np
            @inbounds sum += c_d[nn+1, i, j, k] * Q[interpi[nn], interpj[nn], interpk[nn], 4]
        end
        @inbounds wg = -sum/c_d[1, i, j, k]

        # sum = 0.0
        # for nn = 1:np
        #     @inbounds sum += c_d[nn+1, i, j, k] * Q[interpi[nn], interpj[nn], interpk[nn], 6]
        # end
        @inbounds Tg = Tw #(Tw-sum)/c_d[1, i, j, k]

        ρ = ρmixture(pg, Tg, Yg, thermo)

        ρ_IB = @view ρi[i, j, k, :]
        for n = 1:Nspecs
            @inbounds ρ_IB[n] = Yg[n] * ρ
        end
        @inbounds Q[i, j, k, 7] = InternalEnergy(Tg, ρ_IB, thermo)

        # ug = -Q[interpi[1], interpj[1], interpk[1], 2]
        # vg = -Q[interpi[1], interpj[1], interpk[1], 3]
        # wg = -Q[interpi[1], interpj[1], interpk[1], 4]

        Q[i, j, k, 1] = ρ
        Q[i, j, k, 2] = ug
        Q[i, j, k, 3] = vg
        Q[i, j, k, 4] = wg
        Q[i, j, k, 5] = pg
        Q[i, j, k, 6] = Tg

        @inbounds U[i, j, k, 1] = ρ
        @inbounds U[i, j, k, 2] = ρ * ug
        @inbounds U[i, j, k, 3] = ρ * vg
        @inbounds U[i, j, k, 4] = ρ * wg
        @inbounds U[i, j, k, 5] = Q[i, j, k, 7] + 0.5*ρ*(ug^2+vg^2+wg^2)
    end
    return
end

function init(Q, ρi, ρ, u, v, w, P, T, thermo)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    for n = 1:Nspecs
        @inbounds ρi[i, j, k, n] = 0.0
    end

    rho = ρ
    temp = T
    @inbounds ρi[i, j, k, 2] = rho * 0.233
    @inbounds ρi[i, j, k, 5] = rho * 0.767

    @inbounds rhoi = @view ρi[i, j, k, :]

    @inbounds Q[i, j, k, 1] = rho
    @inbounds Q[i, j, k, 2] = u
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
    T::Float64 = 270.65
    P::Float64 = 79.8
    gas.TPY = T, P, "O2:0.233 N2:0.767"
    ρ::Float64 = gas.density
    u = 1000.0
    w = 0.0
    v = 0.0
    
    @cuda threads=nthreads blocks=nblock init(Q, ρi, ρ, u, v, w, P, T, thermo)
end