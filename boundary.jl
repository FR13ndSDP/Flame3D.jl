function fillGhost(U, NG, Nx, Ny, Nz)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z
    
    if i > Nx+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    gamma::Float64 = 1.4
    T_wall::Float64 = 5323
    Rg::Float64 = 287

    noise::Float64 = rand() * 0.01 * 3757.810529345963
    # Mach 10 inlet
    if i <= 10
        @inbounds U[i, j, k, 1] = 0.035651512619407424
        @inbounds U[i, j, k, 2] = 0.035651512619407424 * 3757.810529345963
        @inbounds U[i, j, k, 3] = 0.0
        @inbounds U[i, j, k, 4] = 0.0
        @inbounds U[i, j, k, 5] = 3596/0.4 + 0.5*0.035651512619407424*3757.810529345963^2
    elseif i > Nx + NG -1
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[Nx+NG-1, j, k, n]
        end
    else
        if j == NG+1 
            @inbounds U[i, j, k, 2] = 0
            @inbounds U[i, j, k, 4] = 0
            if i <= 40 && i >= 30
                @inbounds U[i, j, k, 3] = noise * U[i, j+1, k, 1]
            else
                @inbounds U[i, j, k, 3] = 0
            end
            @inbounds U[i, j, k, 5] = U[i, j+1, k, 5] - 0.5/U[i, j+1, k, 1]*(U[i, j+1, k, 2]^2 + U[i, j+1, k, 3]^2 + U[i, j+1, k, 4]^2)
            @inbounds U[i, j, k, 1] = U[i, j, k, 5] * (gamma-1)/(T_wall * Rg)
        elseif j < NG+1
            p = (gamma-1) * (U[i, 2*NG+2-j, k, 5] - 0.5/U[i, 2*NG+2-j, k, 1]*(U[i, 2*NG+2-j, k, 2]^2 + U[i, 2*NG+2-j, k, 3]^2 + U[i, 2*NG+2-j, k, 4]^2))
            @inbounds U[i, j, k, 1] = p/(Rg * T_wall)
            @inbounds U[i, j, k, 2] = -U[i, 2*NG+2-j, k, 2]/U[i, 2*NG+2-j, k, 1] * U[i, j, k, 1]
            @inbounds U[i, j, k, 3] = -U[i, 2*NG+2-j, k, 3]/U[i, 2*NG+2-j, k, 1] * U[i, j, k, 1]
            @inbounds U[i, j, k, 4] = -U[i, 2*NG+2-j, k, 4]/U[i, 2*NG+2-j, k, 1] * U[i, j, k, 1]
            @inbounds U[i, j, k, 5] = p/(gamma-1) + 0.5/U[i, j, k, 1]*(U[i, j, k, 2]^2 + U[i, j, k, 3]^2 + U[i, j, k, 4]^2)
        elseif j > Ny+NG-1
            for n = 1:Ncons
                @inbounds U[i, j, k, n] = U[i, Ny+NG-1, k, n]
            end
        end
    end

    # periodic
    if k <= NG+1
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, j, Nz-2 + k, n]
        end
    elseif k >= Nz+NG
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, j, k - (Nz-2), n]
        end
    end
    return
end

function fillSpec(ρi, U, NG, Nx, Ny, Nz)
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
        @inbounds ρi[i, j, k, 2] = 0.035651512619407424 * 0.233
        @inbounds ρi[i, j, k, 7] = 0.035651512619407424 * 0.767
    elseif i > Nx + NG -1
        for n = 1:Nspecs
            @inbounds ρi[i, j, k, n] = ρi[Nx+NG-1, j, k, n]
        end
    else
        if j <= NG+1
            for n = 1:Nspecs
                @inbounds ρi[i, j, k, n] = ρi[i, NG+2, k, n]
            end
        elseif j > Ny+NG-1
            for n = 1:Nspecs
                @inbounds ρi[i, j, k, n] = ρi[i, Ny+NG-1, k, n]
            end
        end
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