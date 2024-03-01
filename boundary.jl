function fill_x(Q, U, rank, inlet)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end
    # inlet
    if rank == 0 && i <= NG+1 && j >= NG+1 && j <= Ny+NG
        T_ref = 108.1
        u_ref = 607.177038268082
        ρ_ref = 0.06894018305600735

        ρ = inlet[j-NG, 1] * ρ_ref
        u = inlet[j-NG, 2] * u_ref
        v = inlet[j-NG, 3] * u_ref
        T = inlet[j-NG, 4] * T_ref
        p = ρ * 287 * T

        @inbounds Q[i, j, k, 1] = ρ
        @inbounds Q[i, j, k, 2] = u
        @inbounds Q[i, j, k, 3] = v
        @inbounds Q[i, j, k, 4] = 0.0
        @inbounds Q[i, j, k, 6] = T
        @inbounds Q[i, j, k, 5] = p
        @inbounds Q[i, j, k, 7] = p/0.4

        @inbounds U[i, j, k, 1] = ρ
        @inbounds U[i, j, k, 2] = ρ * u
        @inbounds U[i, j, k, 3] = ρ * v
        @inbounds U[i, j, k, 4] = 0.0
        @inbounds U[i, j, k, 5] = p/0.4 + 0.5*ρ*(u^2+v^2)
    end

    if rank == Nprocs-1 && i > Nxp+NG
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[Nxp+NG, j, k, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[Nxp+NG, j, k, n]
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

    if k <= NG
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, j, k+Nz, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, j, k+Nz, n]
        end
    elseif k > Nz+NG
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, j, k-Nz, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, j, k-Nz, n]
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


    if j == NG+1
        pw = Q[i, j+1, k, 5]
        Tw = 307.0
        ρw = pw/(287 * Tw)
        @inbounds Q[i, j, k, 5] = pw
        @inbounds Q[i, j, k, 1] = ρw
        @inbounds Q[i, j, k, 2] = 0
        @inbounds Q[i, j, k, 3] = 0
        @inbounds Q[i, j, k, 4] = 0
        @inbounds Q[i, j, k, 6] = Tw
        @inbounds Q[i, j, k, 7] = pw/0.4

        @inbounds U[i, j, k, 1] = ρw
        @inbounds U[i, j, k, 2] = 0
        @inbounds U[i, j, k, 3] = 0
        @inbounds U[i, j, k, 4] = 0
        @inbounds U[i, j, k, 5] = pw/0.4

        for l = NG:-1:1
            u = -Q[i, 2*NG+2-l, k, 2]
            v = -Q[i, 2*NG+2-l, k, 3]
            w = -Q[i, 2*NG+2-l, k, 4]
            @inbounds Q[i, l, k, 5] = pw
            @inbounds Q[i, l, k, 1] = ρw
            @inbounds Q[i, l, k, 2] = u
            @inbounds Q[i, l, k, 3] = v
            @inbounds Q[i, l, k, 4] = w
            @inbounds Q[i, l, k, 6] = Tw
            @inbounds Q[i, l, k, 7] = pw/0.4 

            @inbounds U[i, l, k, 1] = ρw
            @inbounds U[i, l, k, 2] = ρw * u
            @inbounds U[i, l, k, 3] = ρw * v
            @inbounds U[i, l, k, 4] = ρw * w
            @inbounds U[i, l, k, 5] = pw/0.4 + 0.5 *Q[i, j, l, 1] * (u^2+v^2+w^2)
        end
    elseif j > Ny+NG
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, Ny+NG, k, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, Ny+NG, k, n]
        end
    end
    return
end

# special treatment on wall
# fill Q and U
function fillGhost(Q, U, rank, inlet)
    @cuda threads=nthreads blocks=nblock fill_x(Q, U, rank, inlet)
    @cuda threads=nthreads blocks=nblock fill_y(Q, U)
    @cuda threads=nthreads blocks=nblock fill_z(Q, U)
end


function init(Q, inlet)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Ny+NG || j < NG+1 || k > Nz+2*NG
        return
    end

    T_ref = 108.1
    u_ref = 607.177038268082
    ρ_ref = 0.06894018305600735

    ρ = inlet[j-NG, 1] * ρ_ref
    u = inlet[j-NG, 2] * u_ref
    v = inlet[j-NG, 3] * u_ref
    T = inlet[j-NG, 4] * T_ref
    p = ρ * 287 * T

    @inbounds Q[i, j, k, 1] = ρ
    @inbounds Q[i, j, k, 2] = u
    @inbounds Q[i, j, k, 3] = v
    @inbounds Q[i, j, k, 4] = 0.0
    @inbounds Q[i, j, k, 6] = T
    @inbounds Q[i, j, k, 5] = p
    @inbounds Q[i, j, k, 7] = p/0.4
    return
end

#initialization on GPU
function initialize(Q, inlet)
    @cuda threads=nthreads blocks=nblock init(Q, inlet)
end