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

    noise::Float64 = rand() * 0.02 * 3757.810529345963
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
            P2 = (gamma-1)*(U[i, j+1, k, 5] - 0.5/U[i, j+1, k, 1]*(U[i, j+1, k, 2]^2 + U[i, j+1, k, 3]^2) + U[i, j+1, k, 4]^2)
            P3 = (gamma-1)*(U[i, j+2, k, 5] - 0.5/U[i, j+2, k, 1]*(U[i, j+2, k, 2]^2 + U[i, j+2, k, 3]^2) + U[i, j+1, k, 4]^2)
            P1 = (4*P2-P3)/3  
            @inbounds U[i, j, k, 1] = P1/(T_wall * Rg)
            @inbounds U[i, j, k, 2] = 0
            @inbounds U[i, j, k, 4] = 0
            if i <= 40 && i >= 30
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
