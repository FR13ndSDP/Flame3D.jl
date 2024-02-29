function fill_x(Q, U, rank, inlet)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end
    # inlet
    if rank == 0 && i <= NG
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[NG+1, j, k, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[NG+1, j, k, n]
        end
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

function fill_y(Q, U)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    if j <= NG
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, 2*NG+2-j, k, n]
        end
        Q[i, j, k, 3] = -Q[i, 2*NG+2-j, k, 3]
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, 2*NG+2-j, k, n]
        end
        U[i, j, k, 3] = -U[i, 2*NG+2-j, k, 3]
    elseif j > Ny+NG
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, 2*Ny+2*NG-j, k, n]
        end
        Q[i, j, k, 3] = -Q[i, 2*NG+2*Ny-j, k, 3]
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, 2*Ny+2*NG-j, k, n]
        end
        U[i, j, k, 3] = -U[i, 2*NG+2*Ny-j, k, 3]
    end
    return
end

function fill_z(Q, U, z3d)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    
    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end


    if k == NG+1
        pw = Q[i, j, k+1, 5]
        Tw = 1000.0
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
            u = -Q[i, j, 2*NG+2-l, 2]
            v = -Q[i, j, 2*NG+2-l, 3]
            w = -Q[i, j, 2*NG+2-l, 4]
            @inbounds Q[i, j, l, 5] = pw
            @inbounds Q[i, j, l, 1] = ρw
            @inbounds Q[i, j, l, 2] = u
            @inbounds Q[i, j, l, 3] = v
            @inbounds Q[i, j, l, 4] = w
            @inbounds Q[i, j, l, 6] = Tw
            @inbounds Q[i, j, l, 7] = pw/0.4 

            @inbounds U[i, j, l, 1] = ρw
            @inbounds U[i, j, l, 2] = ρw * u
            @inbounds U[i, j, l, 3] = ρw * v
            @inbounds U[i, j, l, 4] = ρw * w
            @inbounds U[i, j, l, 5] = pw/0.4 + 0.5 *Q[i, j, l, 1] * (u^2+v^2+w^2)
        end
    elseif k > Nz+NG-1
        zz = z3d[i, j, k]

        if zz > 0
            for n = 1:Nprim
                @inbounds Q[i, j, k, n] = Q[i, j, Nz+NG, n]
            end
            for n = 1:Ncons
                @inbounds U[i, j, k, n] = U[i, j, Nz+NG, n]
            end
        else
            @inbounds Q[i, j, k, 1] = 0.00103
            @inbounds Q[i, j, k, 2] = 4946.98 * cos(40/180*π)
            @inbounds Q[i, j, k, 3] = 0
            @inbounds Q[i, j, k, 4] = 4946.98 * sin(40/180*π)
            @inbounds Q[i, j, k, 5] = 79.8
            @inbounds Q[i, j, k, 6] = 270.65
            @inbounds Q[i, j, k, 7] = 79.8/0.4

            @inbounds U[i, j, k, 1] = 0.00103
            @inbounds U[i, j, k, 2] = 4946.98 * cos(40/180*π) * 0.00103
            @inbounds U[i, j, k, 3] = 0
            @inbounds U[i, j, k, 4] = 4946.98 * sin(40/180*π) * 0.00103
            @inbounds U[i, j, k, 5] = 79.8/0.4 + 0.5*0.00103*4946.98^2
        end
    end
    return
end

# special treatment on wall
# fill Q and U
function fillGhost(Q, U, rank, inlet, z3d)
    @cuda threads=nthreads blocks=nblock fill_x(Q, U, rank, inlet)
    @cuda threads=nthreads blocks=nblock fill_y(Q, U)
    @cuda threads=nthreads blocks=nblock fill_z(Q, U, z3d)
end


function init(Q)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    @inbounds Q[i, j, k, 1] = 0.00103
    @inbounds Q[i, j, k, 2] = 4946.98 * cos(40/180*π)
    @inbounds Q[i, j, k, 3] = 0
    @inbounds Q[i, j, k, 4] = 4946.98 * sin(40/180*π)
    @inbounds Q[i, j, k, 5] = 79.8
    @inbounds Q[i, j, k, 6] = 270.65
    @inbounds Q[i, j, k, 7] = 79.8/0.4
    return
end

#initialization on GPU
function initialize(Q)
    @cuda threads=nthreads blocks=nblock init(Q)
end