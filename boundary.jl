function fill_x(U, NG, Nx, Ny, Nz)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z
    
    if i > Nx+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end
    # Mach 10 inlet
    if i <= 10
        @inbounds U[i, j, k, 1] = 0.03579890492782479
        @inbounds U[i, j, k, 2] = 0.03579890492782479 * 3750.06666607408
        @inbounds U[i, j, k, 3] = 0.0
        @inbounds U[i, j, k, 4] = 0.0
        @inbounds U[i, j, k, 5] = 3596/0.4 + 0.5*0.03579890492782479*3750.06666607408^2
    elseif i > Nx+NG-1
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[Nx+NG-1, j, k, n]
        end
    end
    return
end

function fill_y(U, NG, Nx, Ny, Nz)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z
    
    if i > Nx+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    gamma::Float64 = 1.4
    T_wall::Float64 = 5323
    Rg::Float64 = 287

    noise::Float64 = rand() * 0.05 * 3750.06666607408

    if j == NG+1 
        P2 = (gamma-1)*(U[i, j+1, k, 5] - 0.5/U[i, j+1, k, 1]*(U[i, j+1, k, 2]^2 + U[i, j+1, k, 3]^2) + U[i, j+1, k, 4]^2)
        P3 = (gamma-1)*(U[i, j+2, k, 5] - 0.5/U[i, j+2, k, 1]*(U[i, j+2, k, 2]^2 + U[i, j+2, k, 3]^2) + U[i, j+1, k, 4]^2)
        P1 = (4*P2-P3)/3  
        @inbounds U[i, j, k, 1] = P1/(T_wall * Rg)
        @inbounds U[i, j, k, 2] = 0
        @inbounds U[i, j, k, 4] = 0
        if i <= 150 && i >= 120
            @inbounds U[i, j, k, 3] = noise * U[i, j, k, 1]
        else
            @inbounds U[i, j, k, 3] = 0
        end
        @inbounds U[i, j, k, 5] = P1/(gamma-1) + 0.5/U[i, j, k, 1]*(U[i, j, k, 2]^2 + U[i, j, k, 3]^2 + U[i, j, k, 4]^2)
        
        for l = 1:NG
            P2 = (gamma-1)*(U[i, j-l+1, k, 5] - 0.5/U[i, j-l+1, k, 1]*(U[i, j-l+1, k, 2]^2 + U[i, j-l+1, k, 3]^2) + U[i, j-l+1, k, 4]^2)
            P3 = (gamma-1)*(U[i, j-l+2, k, 5] - 0.5/U[i, j-l+2, k, 1]*(U[i, j-l+2, k, 2]^2 + U[i, j-l+2, k, 3]^2) + U[i, j-l+1, k, 4]^2)
            P1 = (4*P2-P3)/3  
            Uin = U[i, 2*NG+2-(j-l), k, 2]/U[i, 2*NG+2-(j-l), k, 1]
            Vin = U[i, 2*NG+2-(j-l), k, 3]/U[i, 2*NG+2-(j-l), k, 1]
            Win = U[i, 2*NG+2-(j-l), k, 4]/U[i, 2*NG+2-(j-l), k, 1]
            @inbounds U[i, j-l, k, 1] = P1/(Rg * T_wall)
            @inbounds U[i, j-l, k, 2] = -U[i, j-l, k, 1]*Uin
            @inbounds U[i, j-l, k, 3] = -U[i, j-l, k, 1]*Vin
            @inbounds U[i, j-l, k, 4] = -U[i, j-l, k, 1]*Win
            @inbounds U[i, j-l, k, 5] = P1/(gamma-1) + 0.5/U[i, j-l, k, 1]*(U[i, j-l, k, 2]^2 + U[i, j-l, k, 3]^2 + U[i, j-l, k, 4]^2)
        end
    elseif j > Ny+NG-1
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, Ny+NG-1, k, n]
        end
    end
    return
end

function fill_z(U, NG, Nx, Ny, Nz)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z
    
    if i > Nx+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    if k <= NG+1
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, j, Nz-2 + k, n]
        end
    elseif k > Nz+NG-1
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, j, k - (Nz-2), n]
        end
    end
    return
end

function fill_x_s(ρi, NG, Nx, Ny, Nz)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z
    
    if i > Nx+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    if i <= 10
        for n = 1:Nspecs
            @inbounds ρi[i, j, k, n] = 0
        end
        @inbounds ρi[i, j, k, 2] = 0.03579890492782479 * 0.233
        @inbounds ρi[i, j, k, 5] = 0.03579890492782479 * 0.767
    elseif i > Nx+NG-1
        for n = 1:Nspecs
            @inbounds ρi[i, j, k, n] = ρi[Nx+NG-1, j, k, n]
        end
    end
    return
end

function fill_y_s(ρi, U, NG, Nx, Ny, Nz)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z
    
    if i > Nx+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    if j <= NG+1
        for n = 1:Nspecs
            @inbounds Yi = ρi[i, NG+2, k, n]/U[i, NG+2, k, 1]
            @inbounds ρi[i, j, k, n] = Yi * U[i, j, k, 1]
        end
    elseif j > Ny+NG-1
        for n = 1:Nspecs
            @inbounds ρi[i, j, k, n] = ρi[i, Ny+NG-1, k, n]
        end
    end
    return
end

function fill_z_s(ρi, NG, Nx, Ny, Nz)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z
    
    if i > Nx+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    # periodic
    if k <= NG+1
        for n = 1:Nspecs
            @inbounds ρi[i, j, k, n] = ρi[i, j, Nz-2 + k, n]
        end
    elseif k >= Nz+NG
        for n = 1:Nspecs
            @inbounds ρi[i, j, k, n] = ρi[i, j, k - (Nz-2), n]
        end
    end
    return
end

function fillGhost(U, NG, Nx, Ny, Nz)
    @cuda threads=nthreads blocks=nblock fill_y(U, NG, Nx, Ny, Nz)
    @cuda threads=nthreads blocks=nblock fill_x(U, NG, Nx, Ny, Nz)
    @cuda threads=nthreads blocks=nblock fill_z(U, NG, Nx, Ny, Nz)
end

function fillSpec(ρi, U, NG, Nx, Ny, Nz)
    @cuda threads=nthreads blocks=nblock fill_y_s(ρi, U, NG, Nx, Ny, Nz)
    @cuda threads=nthreads blocks=nblock fill_x_s(ρi, NG, Nx, Ny, Nz)
    @cuda threads=nthreads blocks=nblock fill_z_s(ρi, NG, Nx, Ny, Nz)
end