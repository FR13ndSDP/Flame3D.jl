function fill_x(U, Q, prof, ρi, thermo)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z
    
    if i > Nx+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end
    # Mach 10 inlet
    if i <= NG+1

        @inbounds Q[i, j, k, 1] = prof[j, 1]
        @inbounds Q[i, j, k, 2] = prof[j, 2]
        @inbounds Q[i, j, k, 3] = prof[j, 3]
        @inbounds Q[i, j, k, 4] = 0
        @inbounds Q[i, j, k, 5] = prof[j, 5]
        @inbounds Q[i, j, k, 6] = prof[j, 6]

        @inbounds rhoi = @view ρi[i, j, k, :]
        ei = InternalEnergy(prof[j, 6], rhoi, thermo)
        @inbounds U[i, j, k, 1] = prof[j, 1]
        @inbounds U[i, j, k, 2] = prof[j, 1] * prof[j, 2]
        @inbounds U[i, j, k, 3] = prof[j, 1] * prof[j, 3]
        @inbounds U[i, j, k, 4] = 0
        @inbounds U[i, j, k, 5] = ei + 0.5*prof[j, 1]*(prof[j, 2]^2 + prof[j, 3]^2)
    elseif i > Nx+NG-1
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[Nx+NG-1, j, k, n]
        end

        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[Nx+NG-1, j, k, n]
        end
    end
    return
end

function fill_y(U, Q, ρi, Yi, thermo)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z
    
    if i > Nx+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    T_wall::Float64 = 5323

    noise::Float64 = (2*rand()-1) * 0.05 * 3175.3718050448897

    if j == NG+1
        @inbounds P2 = Q[i, j+1, k, 5]
        @inbounds P3 = Q[i, j+2, k, 5]
        P1 = (4*P2-P3)/3  
        @inbounds Y = @view Yi[i, NG+2, k, :]
        rho = ρmixture(P1, T_wall, Y, thermo)

        for l = 0:NG
            for n = 1:Nspecs
                @inbounds Yi[i, j-l, k, n] = Yi[i, NG+2, k, n]
                @inbounds ρi[i, j-l, k, n] = Yi[i, NG+2, k, n] * rho
            end
        end

        @inbounds Q[i, j, k, 1] = rho
        @inbounds Q[i, j, k, 2] = 0
        @inbounds Q[i, j, k, 4] = 0
        @inbounds Q[i, j, k, 5] = P1
        @inbounds Q[i, j, k, 6] = T_wall

        @inbounds d = @view ρi[i, j, k, :]
        ei = InternalEnergy(T_wall, d, thermo)

        @inbounds U[i, j, k, 1] = rho
        @inbounds U[i, j, k, 2] = 0
        @inbounds U[i, j, k, 4] = 0
        if i <= 30 && i >= 20
            @inbounds Q[i, j, k, 3] = noise
            @inbounds U[i, j, k, 3] = noise * rho
        else
            @inbounds Q[i, j, k, 3] = 0
            @inbounds U[i, j, k, 3] = 0
        end
        @inbounds U[i, j, k, 5] = ei + 0.5*rho*Q[i, j, k, 3]^2
        
        for l = 1:NG
            @inbounds Uin = -Q[i, 2*NG+2-(j-l), k, 2]
            @inbounds Vin = -Q[i, 2*NG+2-(j-l), k, 3]
            @inbounds Win = -Q[i, 2*NG+2-(j-l), k, 4]
            @inbounds Q[i, j-l, k, 1] = rho
            @inbounds Q[i, j-l, k, 2] = Uin
            @inbounds Q[i, j-l, k, 3] = Vin
            @inbounds Q[i, j-l, k, 4] = Win
            @inbounds Q[i, j-l, k, 5] = P1
            @inbounds Q[i, j-l, k, 6] = T_wall
            
            @inbounds U[i, j-l, k, 1] = rho
            @inbounds U[i, j-l, k, 2] = rho*Uin
            @inbounds U[i, j-l, k, 3] = rho*Vin
            @inbounds U[i, j-l, k, 4] = rho*Win
            @inbounds U[i, j-l, k, 5] = ei + 0.5*rho*(Uin^2+Vin^2+Win^2)
        end
    elseif j > Ny+NG-1
        for n = 1:Nspecs
            @inbounds ρi[i, j, k, n] = ρi[i, Ny+NG-1, k, n]
        end
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, Ny+NG-1, k, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, Ny+NG-1, k, n]
        end
    end
    return
end

function fill_z(U, Q)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z
    
    if i > Nx+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    if k <= NG+1
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, j, Nz-2 + k, n]
        end

        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, j, Nz-2 + k, n]
        end
    elseif k > Nz+NG-1
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, j, k - (Nz-2), n]
        end

        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, j, k - (Nz-2), n]
        end
    end
    return
end

function fill_x_s(ρi, Yi, Q)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z
    
    if i > Nx+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    if i <= 10
        for n = 1:Nspecs
            @inbounds Yi[i, j, k, n] = 0
            @inbounds ρi[i, j, k, n] = 0
        end
        @inbounds Yi[i, j, k, 2] = 0.233
        @inbounds Yi[i, j, k, 5] = 0.767
        @inbounds ρi[i, j, k, 2] = Yi[i, j, k, 2] * Q[i, j, k, 1]
        @inbounds ρi[i, j, k, 5] = Yi[i, j, k, 5] * Q[i, j, k, 1]
    elseif i > Nx+NG-1
        for n = 1:Nspecs
            @inbounds Yi[i, j, k, n] = Yi[Nx+NG-1, j, k, n]
            @inbounds ρi[i, j, k, n] = Yi[i, j, k, n] * Q[i, j, k, 1]
        end
    end
    return
end

function fill_z_s(ρi, Yi, Q)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z
    
    if i > Nx+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    # periodic
    if k <= NG+1
        for n = 1:Nspecs
            @inbounds Yi[i, j, k, n] = Yi[i, j, Nz-2 + k, n]
            @inbounds ρi[i, j, k, n] = Yi[i, j, k, n] * Q[i, j, k, 1]
        end
    elseif k >= Nz+NG
        for n = 1:Nspecs
            @inbounds Yi[i, j, k, n] = Yi[i, j, k - (Nz-2), n]
            @inbounds ρi[i, j, k, n] = Yi[i, j, k, n] * Q[i, j, k, 1]
        end
    end
    return
end

# special treatment on wall
function fillGhost(U, Q, ρi, Yi, thermo, prof)
    @cuda threads=nthreads blocks=nblock fill_y(U, Q, ρi, Yi, thermo)
    @cuda threads=nthreads blocks=nblock fill_x(U, Q, prof, ρi, thermo)
    @cuda threads=nthreads blocks=nblock fill_z(U, Q)
end

# only in two trival directions
function fillSpec(ρi, Yi, Q)
    @cuda threads=nthreads blocks=nblock fill_x_s(ρi, Yi, Q)
    @cuda threads=nthreads blocks=nblock fill_z_s(ρi, Yi, Q)
end