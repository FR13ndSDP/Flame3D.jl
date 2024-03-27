function viscousFlux_x(Fv_x, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, μratio_x)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG+1 || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    c23::Float32 = 2/3f0
    c12::Float32 = 1/12f0

    @inbounds ∂ξ∂x = (dξdx[i-1, j, k] + dξdx[i, j, k]) * 0.5f0
    @inbounds ∂ξ∂y = (dξdy[i-1, j, k] + dξdy[i, j, k]) * 0.5f0
    @inbounds ∂ξ∂z = (dξdz[i-1, j, k] + dξdz[i, j, k]) * 0.5f0
    @inbounds ∂η∂x = (dηdx[i-1, j, k] + dηdx[i, j, k]) * 0.5f0
    @inbounds ∂η∂y = (dηdy[i-1, j, k] + dηdy[i, j, k]) * 0.5f0
    @inbounds ∂η∂z = (dηdz[i-1, j, k] + dηdz[i, j, k]) * 0.5f0
    @inbounds ∂ζ∂x = (dζdx[i-1, j, k] + dζdx[i, j, k]) * 0.5f0
    @inbounds ∂ζ∂y = (dζdy[i-1, j, k] + dζdy[i, j, k]) * 0.5f0
    @inbounds ∂ζ∂z = (dζdz[i-1, j, k] + dζdz[i, j, k]) * 0.5f0

    @inbounds Jac = (J[i-1, j, k] + J[i, j, k]) * 0.5f0
    @inbounds T = (Q[i-1, j, k, 6] + Q[i, j, k, 6]) * 0.5f0
    @inbounds μi =  1.458f-6*T*sqrt(T)/(T+110.4f0)
    @inbounds λi =  1004.5f0*μi/0.7f0

    @inbounds ∂u∂ξ = 1.25f0*(Q[i, j, k, 2] - Q[i-1, j, k, 2]) - c12*(Q[i+1, j, k, 2] - Q[i-2, j, k, 2])
    @inbounds ∂v∂ξ = 1.25f0*(Q[i, j, k, 3] - Q[i-1, j, k, 3]) - c12*(Q[i+1, j, k, 3] - Q[i-2, j, k, 3])
    @inbounds ∂w∂ξ = 1.25f0*(Q[i, j, k, 4] - Q[i-1, j, k, 4]) - c12*(Q[i+1, j, k, 4] - Q[i-2, j, k, 4])
    @inbounds ∂T∂ξ = 1.25f0*(Q[i, j, k, 6] - Q[i-1, j, k, 6]) - c12*(Q[i+1, j, k, 6] - Q[i-2, j, k, 6])

    @inbounds ∂u∂η = 0.5f0*(c23*(Q[i, j+1, k, 2] + Q[i-1, j+1, k, 2] - Q[i, j-1, k, 2] - Q[i-1, j-1, k, 2]) -
                          c12*(Q[i, j+2, k, 2] + Q[i-1, j+2, k, 2] - Q[i, j-2, k, 2] - Q[i-1, j-2, k, 2]))
    @inbounds ∂v∂η = 0.5f0*(c23*(Q[i, j+1, k, 3] + Q[i-1, j+1, k, 3] - Q[i, j-1, k, 3] - Q[i-1, j-1, k, 3]) -
                          c12*(Q[i, j+2, k, 3] + Q[i-1, j+2, k, 3] - Q[i, j-2, k, 3] - Q[i-1, j-2, k, 3]))
    @inbounds ∂w∂η = 0.5f0*(c23*(Q[i, j+1, k, 4] + Q[i-1, j+1, k, 4] - Q[i, j-1, k, 4] - Q[i-1, j-1, k, 4]) -
                          c12*(Q[i, j+2, k, 4] + Q[i-1, j+2, k, 4] - Q[i, j-2, k, 4] - Q[i-1, j-2, k, 4]))
    @inbounds ∂T∂η = 0.5f0*(c23*(Q[i, j+1, k, 6] + Q[i-1, j+1, k, 6] - Q[i, j-1, k, 6] - Q[i-1, j-1, k, 6]) -
                          c12*(Q[i, j+2, k, 6] + Q[i-1, j+2, k, 6] - Q[i, j-2, k, 6] - Q[i-1, j-2, k, 6]))

    @inbounds ∂u∂ζ = 0.5f0*(c23*(Q[i, j, k+1, 2] + Q[i-1, j, k+1, 2] - Q[i, j, k-1, 2] - Q[i-1, j, k-1, 2]) - 
                          c12*(Q[i, j, k+2, 2] + Q[i-1, j, k+2, 2] - Q[i, j, k-2, 2] - Q[i-1, j, k-2, 2]))
    @inbounds ∂v∂ζ = 0.5f0*(c23*(Q[i, j, k+1, 3] + Q[i-1, j, k+1, 3] - Q[i, j, k-1, 3] - Q[i-1, j, k-1, 3]) - 
                          c12*(Q[i, j, k+2, 3] + Q[i-1, j, k+2, 3] - Q[i, j, k-2, 3] - Q[i-1, j, k-2, 3]))
    @inbounds ∂w∂ζ = 0.5f0*(c23*(Q[i, j, k+1, 4] + Q[i-1, j, k+1, 4] - Q[i, j, k-1, 4] - Q[i-1, j, k-1, 4]) - 
                          c12*(Q[i, j, k+2, 4] + Q[i-1, j, k+2, 4] - Q[i, j, k-2, 4] - Q[i-1, j, k-2, 4]))
    @inbounds ∂T∂ζ = 0.5f0*(c23*(Q[i, j, k+1, 6] + Q[i-1, j, k+1, 6] - Q[i, j, k-1, 6] - Q[i-1, j, k-1, 6]) - 
                          c12*(Q[i, j, k+2, 6] + Q[i-1, j, k+2, 6] - Q[i, j, k-2, 6] - Q[i-1, j, k-2, 6]))

    @inbounds u = (Q[i-1, j, k, 2] + Q[i, j, k, 2]) * 0.5f0
    @inbounds v = (Q[i-1, j, k, 3] + Q[i, j, k, 3]) * 0.5f0
    @inbounds w = (Q[i-1, j, k, 4] + Q[i, j, k, 4]) * 0.5f0

    dudx = (∂u∂ξ * ∂ξ∂x + ∂u∂η * ∂η∂x + ∂u∂ζ * ∂ζ∂x) * Jac
    dudy = (∂u∂ξ * ∂ξ∂y + ∂u∂η * ∂η∂y + ∂u∂ζ * ∂ζ∂y) * Jac
    dudz = (∂u∂ξ * ∂ξ∂z + ∂u∂η * ∂η∂z + ∂u∂ζ * ∂ζ∂z) * Jac

    dvdx = (∂v∂ξ * ∂ξ∂x + ∂v∂η * ∂η∂x + ∂v∂ζ * ∂ζ∂x) * Jac
    dvdy = (∂v∂ξ * ∂ξ∂y + ∂v∂η * ∂η∂y + ∂v∂ζ * ∂ζ∂y) * Jac
    dvdz = (∂v∂ξ * ∂ξ∂z + ∂v∂η * ∂η∂z + ∂v∂ζ * ∂ζ∂z) * Jac

    dwdx = (∂w∂ξ * ∂ξ∂x + ∂w∂η * ∂η∂x + ∂w∂ζ * ∂ζ∂x) * Jac
    dwdy = (∂w∂ξ * ∂ξ∂y + ∂w∂η * ∂η∂y + ∂w∂ζ * ∂ζ∂y) * Jac
    dwdz = (∂w∂ξ * ∂ξ∂z + ∂w∂η * ∂η∂z + ∂w∂ζ * ∂ζ∂z) * Jac

    dTdx = (∂T∂ξ * ∂ξ∂x + ∂T∂η * ∂η∂x + ∂T∂ζ * ∂ζ∂x) * Jac
    dTdy = (∂T∂ξ * ∂ξ∂y + ∂T∂η * ∂η∂y + ∂T∂ζ * ∂ζ∂y) * Jac
    dTdz = (∂T∂ξ * ∂ξ∂z + ∂T∂η * ∂η∂z + ∂T∂ζ * ∂ζ∂z) * Jac

    div = dudx + dvdy + dwdz

    if LES_smag
        Cs = 0.1f0
        Prt = 0.9f0
        @inbounds ρ = 0.5f0 * (Q[i, j, k, 1] + Q[i-1, j, k, 1])

        @fastmath Sijmag = sqrt(2*(dudx^2 + dvdy^2 + dwdz^2 + 
                               2*((0.5f0*(dudy+dvdx))^2 + (0.5f0*(dudz+dwdx))^2 +(0.5f0*(dvdz+dwdy))^2))) # √2|sij|
      
        @fastmath μt = ρ * (Cs/Jac^(1/3f0))^2 * Sijmag #ρ(csΔ)^2 * Sijmag

        λt = 1004.5f0 * μt / Prt # cp = Rg*γ/(γ-1)
        μratio_x[i-NG, j-NG, k-NG] = μt/μi

        μi += μt
        λi += λt
    elseif LES_wale
        Cw = 0.325f0
        Prt = 0.9f0
        @inbounds ρ = 0.5f0 * (Q[i, j, k, 1] + Q[i-1, j, k, 1])

        @fastmath S = sqrt(dudx^2 + dvdy^2 + dwdz^2 + 
        2*((0.5f0*(dudy+dvdx))^2 + (0.5f0*(dudz+dwdx))^2 +(0.5f0*(dvdz+dwdy))^2))

        Sd11 = dudx*dudx + dudy*dvdx + dudz*dwdx
        Sd22 = dvdx*dudy + dvdy*dvdy + dvdz*dwdy
        Sd33 = dwdx*dudz + dwdy*dvdz + dwdz*dwdz
        trSd = 1/3f0*(Sd11 + Sd22 + Sd33)
        Sd11 -= trSd
        Sd22 -= trSd
        Sd33 -= trSd
        Sd12 = 0.5f0*(dudx*dvdx + dvdx*dvdy + dwdx*dvdz + dudy*dudx + dvdy*dudy + dwdy*dudz)
        Sd13 = 0.5f0*(dudx*dwdx + dvdx*dwdy + dwdx*dwdz + dudz*dudx + dvdz*dudy + dwdz*dudz)
        Sd23 = 0.5f0*(dudy*dwdx + dvdy*dwdy + dwdy*dwdz + dudz*dvdx + dvdz*dvdy + dwdz*dvdz)
        @fastmath Sd = sqrt(Sd11^2 + Sd22^2 + Sd33^2 + 2 * (Sd12^2 + Sd13^2 + Sd23^2))
        @fastmath D = Sd^3/(S^5 + Sd^2.5f0)
        @fastmath μt = min(ρ * (Cw/Jac^(1/3f0))^2 * D, 6*μi)
      
        λt = 1004.5f0 * μt / Prt # cp = Rg*γ/(γ-1)
        μratio_x[i-NG, j-NG, k-NG] = μt/μi

        μi += μt
        λi += λt
    end

    τ11 = μi*(2*dudx - c23*div)
    τ12 = μi*(dudy + dvdx)
    τ13 = μi*(dudz + dwdx)
    τ22 = μi*(2*dvdy - c23*div)
    τ23 = μi*(dwdy + dvdz)
    τ33 = μi*(2*dwdz - c23*div)

    @inbounds E1 = u * τ11 + v * τ12 + w * τ13 + λi * dTdx
    @inbounds E2 = u * τ12 + v * τ22 + w * τ23 + λi * dTdy
    @inbounds E3 = u * τ13 + v * τ23 + w * τ33 + λi * dTdz

    @inbounds Fv_x[i-NG, j-NG, k-NG, 1] = ∂ξ∂x * τ11 + ∂ξ∂y * τ12 + ∂ξ∂z * τ13
    @inbounds Fv_x[i-NG, j-NG, k-NG, 2] = ∂ξ∂x * τ12 + ∂ξ∂y * τ22 + ∂ξ∂z * τ23
    @inbounds Fv_x[i-NG, j-NG, k-NG, 3] = ∂ξ∂x * τ13 + ∂ξ∂y * τ23 + ∂ξ∂z * τ33
    @inbounds Fv_x[i-NG, j-NG, k-NG, 4] = ∂ξ∂x * E1 + ∂ξ∂y * E2 + ∂ξ∂z * E3
    return
end

function viscousFlux_y(Fv_y, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, μratio_y)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Nyp+NG+1 || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    c23::Float32 = 2/3f0
    c12::Float32 = 1/12f0

    @inbounds ∂ξ∂x = (dξdx[i, j-1, k] + dξdx[i, j, k]) * 0.5f0
    @inbounds ∂ξ∂y = (dξdy[i, j-1, k] + dξdy[i, j, k]) * 0.5f0
    @inbounds ∂ξ∂z = (dξdz[i, j-1, k] + dξdz[i, j, k]) * 0.5f0
    @inbounds ∂η∂x = (dηdx[i, j-1, k] + dηdx[i, j, k]) * 0.5f0
    @inbounds ∂η∂y = (dηdy[i, j-1, k] + dηdy[i, j, k]) * 0.5f0
    @inbounds ∂η∂z = (dηdz[i, j-1, k] + dηdz[i, j, k]) * 0.5f0
    @inbounds ∂ζ∂x = (dζdx[i, j-1, k] + dζdx[i, j, k]) * 0.5f0
    @inbounds ∂ζ∂y = (dζdy[i, j-1, k] + dζdy[i, j, k]) * 0.5f0
    @inbounds ∂ζ∂z = (dζdz[i, j-1, k] + dζdz[i, j, k]) * 0.5f0

    @inbounds Jac = (J[i, j-1, k] + J[i, j, k]) * 0.5f0
    @inbounds T = (Q[i, j-1, k, 6] + Q[i, j, k, 6]) * 0.5f0
    @inbounds μi =  1.458f-6*T*sqrt(T)/(T+110.4f0)
    @inbounds λi =  1004.5f0*μi/0.7f0

    @inbounds ∂u∂ξ = 0.5f0*(c23*(Q[i+1, j, k, 2] + Q[i+1, j-1, k, 2] - Q[i-1, j, k, 2] - Q[i-1, j-1, k, 2]) -
                          c12*(Q[i+2, j, k, 2] + Q[i+2, j-1, k, 2] - Q[i-2, j, k, 2] - Q[i-2, j-1, k, 2]))
    @inbounds ∂v∂ξ = 0.5f0*(c23*(Q[i+1, j, k, 3] + Q[i+1, j-1, k, 3] - Q[i-1, j, k, 3] - Q[i-1, j-1, k, 3]) -
                          c12*(Q[i+2, j, k, 3] + Q[i+2, j-1, k, 3] - Q[i-2, j, k, 3] - Q[i-2, j-1, k, 3]))
    @inbounds ∂w∂ξ = 0.5f0*(c23*(Q[i+1, j, k, 4] + Q[i+1, j-1, k, 4] - Q[i-1, j, k, 4] - Q[i-1, j-1, k, 4]) -
                          c12*(Q[i+2, j, k, 4] + Q[i+2, j-1, k, 4] - Q[i-2, j, k, 4] - Q[i-2, j-1, k, 4]))
    @inbounds ∂T∂ξ = 0.5f0*(c23*(Q[i+1, j, k, 6] + Q[i+1, j-1, k, 6] - Q[i-1, j, k, 6] - Q[i-1, j-1, k, 6]) -
                          c12*(Q[i+2, j, k, 6] + Q[i+2, j-1, k, 6] - Q[i-2, j, k, 6] - Q[i-2, j-1, k, 6]))

    @inbounds ∂u∂η = 1.25f0*(Q[i, j, k, 2] - Q[i, j-1, k, 2]) - c12*(Q[i, j+1, k, 2] - Q[i, j-2, k, 2])
    @inbounds ∂v∂η = 1.25f0*(Q[i, j, k, 3] - Q[i, j-1, k, 3]) - c12*(Q[i, j+1, k, 3] - Q[i, j-2, k, 3])
    @inbounds ∂w∂η = 1.25f0*(Q[i, j, k, 4] - Q[i, j-1, k, 4]) - c12*(Q[i, j+1, k, 4] - Q[i, j-2, k, 4])
    @inbounds ∂T∂η = 1.25f0*(Q[i, j, k, 6] - Q[i, j-1, k, 6]) - c12*(Q[i, j+1, k, 6] - Q[i, j-2, k, 6])

    @inbounds ∂u∂ζ = 0.5f0*(c23*(Q[i, j, k+1, 2] + Q[i, j-1, k+1, 2] - Q[i, j, k-1, 2] - Q[i, j-1, k-1, 2]) -
                          c12*(Q[i, j, k+2, 2] + Q[i, j-1, k+2, 2] - Q[i, j, k-2, 2] - Q[i, j-1, k-2, 2])) 
    @inbounds ∂v∂ζ = 0.5f0*(c23*(Q[i, j, k+1, 3] + Q[i, j-1, k+1, 3] - Q[i, j, k-1, 3] - Q[i, j-1, k-1, 3]) - 
                          c12*(Q[i, j, k+2, 3] + Q[i, j-1, k+2, 3] - Q[i, j, k-2, 3] - Q[i, j-1, k-2, 3])) 
    @inbounds ∂w∂ζ = 0.5f0*(c23*(Q[i, j, k+1, 4] + Q[i, j-1, k+1, 4] - Q[i, j, k-1, 4] - Q[i, j-1, k-1, 4]) - 
                          c12*(Q[i, j, k+2, 4] + Q[i, j-1, k+2, 4] - Q[i, j, k-2, 4] - Q[i, j-1, k-2, 4])) 
    @inbounds ∂T∂ζ = 0.5f0*(c23*(Q[i, j, k+1, 6] + Q[i, j-1, k+1, 6] - Q[i, j, k-1, 6] - Q[i, j-1, k-1, 6]) - 
                          c12*(Q[i, j, k+2, 6] + Q[i, j-1, k+2, 6] - Q[i, j, k-2, 6] - Q[i, j-1, k-2, 6])) 

    @inbounds u = (Q[i, j-1, k, 2] + Q[i, j, k, 2]) * 0.5f0
    @inbounds v = (Q[i, j-1, k, 3] + Q[i, j, k, 3]) * 0.5f0
    @inbounds w = (Q[i, j-1, k, 4] + Q[i, j, k, 4]) * 0.5f0

    dudx = (∂u∂ξ * ∂ξ∂x + ∂u∂η * ∂η∂x + ∂u∂ζ * ∂ζ∂x) * Jac
    dudy = (∂u∂ξ * ∂ξ∂y + ∂u∂η * ∂η∂y + ∂u∂ζ * ∂ζ∂y) * Jac
    dudz = (∂u∂ξ * ∂ξ∂z + ∂u∂η * ∂η∂z + ∂u∂ζ * ∂ζ∂z) * Jac

    dvdx = (∂v∂ξ * ∂ξ∂x + ∂v∂η * ∂η∂x + ∂v∂ζ * ∂ζ∂x) * Jac
    dvdy = (∂v∂ξ * ∂ξ∂y + ∂v∂η * ∂η∂y + ∂v∂ζ * ∂ζ∂y) * Jac
    dvdz = (∂v∂ξ * ∂ξ∂z + ∂v∂η * ∂η∂z + ∂v∂ζ * ∂ζ∂z) * Jac

    dwdx = (∂w∂ξ * ∂ξ∂x + ∂w∂η * ∂η∂x + ∂w∂ζ * ∂ζ∂x) * Jac
    dwdy = (∂w∂ξ * ∂ξ∂y + ∂w∂η * ∂η∂y + ∂w∂ζ * ∂ζ∂y) * Jac
    dwdz = (∂w∂ξ * ∂ξ∂z + ∂w∂η * ∂η∂z + ∂w∂ζ * ∂ζ∂z) * Jac

    dTdx = (∂T∂ξ * ∂ξ∂x + ∂T∂η * ∂η∂x + ∂T∂ζ * ∂ζ∂x) * Jac
    dTdy = (∂T∂ξ * ∂ξ∂y + ∂T∂η * ∂η∂y + ∂T∂ζ * ∂ζ∂y) * Jac
    dTdz = (∂T∂ξ * ∂ξ∂z + ∂T∂η * ∂η∂z + ∂T∂ζ * ∂ζ∂z) * Jac

    div = dudx + dvdy + dwdz

    if LES_smag
        Cs = 0.1f0
        Prt = 0.9f0
        @inbounds ρ = 0.5f0 * (Q[i, j, k, 1] + Q[i, j-1, k, 1])

        @fastmath Sijmag = sqrt(2*(dudx^2 + dvdy^2 + dwdz^2 + 
                               2*((0.5f0*(dudy+dvdx))^2 + (0.5f0*(dudz+dwdx))^2 +(0.5f0*(dvdz+dwdy))^2))) # √2|sij|
      
        @fastmath μt = ρ * (Cs/Jac^(1/3f0))^2 * Sijmag #ρ(csΔ)^2 * Sijmag
      
        λt = 1004.5f0 * μt / Prt # cp = Rg*γ/(γ-1)
        μratio_y[i-NG, j-NG, k-NG] = μt/μi

        μi += μt
        λi += λt
    elseif LES_wale
        Cw = 0.325f0
        Prt = 0.9f0
        @inbounds ρ = 0.5f0 * (Q[i, j, k, 1] + Q[i, j-1, k, 1])

        @fastmath S = sqrt(dudx^2 + dvdy^2 + dwdz^2 + 
        2*((0.5f0*(dudy+dvdx))^2 + (0.5f0*(dudz+dwdx))^2 +(0.5f0*(dvdz+dwdy))^2))

        Sd11 = dudx*dudx + dudy*dvdx + dudz*dwdx
        Sd22 = dvdx*dudy + dvdy*dvdy + dvdz*dwdy
        Sd33 = dwdx*dudz + dwdy*dvdz + dwdz*dwdz
        trSd = 1/3f0*(Sd11 + Sd22 + Sd33)
        Sd11 -= trSd
        Sd22 -= trSd
        Sd33 -= trSd
        Sd12 = 0.5f0*(dudx*dvdx + dvdx*dvdy + dwdx*dvdz + dudy*dudx + dvdy*dudy + dwdy*dudz)
        Sd13 = 0.5f0*(dudx*dwdx + dvdx*dwdy + dwdx*dwdz + dudz*dudx + dvdz*dudy + dwdz*dudz)
        Sd23 = 0.5f0*(dudy*dwdx + dvdy*dwdy + dwdy*dwdz + dudz*dvdx + dvdz*dvdy + dwdz*dvdz)
        @fastmath Sd = sqrt(Sd11^2 + Sd22^2 + Sd33^2 + 2 * (Sd12^2 + Sd13^2 + Sd23^2))
        @fastmath D = Sd^3/(S^5 + Sd^2.5f0)
        @fastmath μt = min(ρ * (Cw/Jac^(1/3f0))^2 * D, 6*μi)
      
        λt = 1004.5f0 * μt / Prt # cp = Rg*γ/(γ-1)
        μratio_y[i-NG, j-NG, k-NG] = μt/μi

        μi += μt
        λi += λt
    end

    τ11 = μi*(2*dudx - c23*div)
    τ12 = μi*(dudy + dvdx)
    τ13 = μi*(dudz + dwdx)
    τ22 = μi*(2*dvdy - c23*div)
    τ23 = μi*(dwdy + dvdz)
    τ33 = μi*(2*dwdz - c23*div)

    @inbounds E1 = u * τ11 + v * τ12 + w * τ13 + λi * dTdx
    @inbounds E2 = u * τ12 + v * τ22 + w * τ23 + λi * dTdy
    @inbounds E3 = u * τ13 + v * τ23 + w * τ33 + λi * dTdz

    @inbounds Fv_y[i-NG, j-NG, k-NG, 1] = ∂η∂x * τ11 + ∂η∂y * τ12 + ∂η∂z * τ13
    @inbounds Fv_y[i-NG, j-NG, k-NG, 2] = ∂η∂x * τ12 + ∂η∂y * τ22 + ∂η∂z * τ23
    @inbounds Fv_y[i-NG, j-NG, k-NG, 3] = ∂η∂x * τ13 + ∂η∂y * τ23 + ∂η∂z * τ33
    @inbounds Fv_y[i-NG, j-NG, k-NG, 4] = ∂η∂x * E1 + ∂η∂y * E2 + ∂η∂z * E3
    return
end

function viscousFlux_z(Fv_z, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, μratio_z)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG+1 || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    c23::Float32 = 2/3f0
    c12::Float32 = 1/12f0

    @inbounds ∂ξ∂x = (dξdx[i, j, k-1] + dξdx[i, j, k]) * 0.5f0
    @inbounds ∂ξ∂y = (dξdy[i, j, k-1] + dξdy[i, j, k]) * 0.5f0
    @inbounds ∂ξ∂z = (dξdz[i, j, k-1] + dξdz[i, j, k]) * 0.5f0
    @inbounds ∂η∂x = (dηdx[i, j, k-1] + dηdx[i, j, k]) * 0.5f0
    @inbounds ∂η∂y = (dηdy[i, j, k-1] + dηdy[i, j, k]) * 0.5f0
    @inbounds ∂η∂z = (dηdz[i, j, k-1] + dηdz[i, j, k]) * 0.5f0
    @inbounds ∂ζ∂x = (dζdx[i, j, k-1] + dζdx[i, j, k]) * 0.5f0
    @inbounds ∂ζ∂y = (dζdy[i, j, k-1] + dζdy[i, j, k]) * 0.5f0
    @inbounds ∂ζ∂z = (dζdz[i, j, k-1] + dζdz[i, j, k]) * 0.5f0

    @inbounds Jac = (J[i, j, k-1] + J[i, j, k]) * 0.5f0
    @inbounds T = (Q[i, j, k-1, 6] + Q[i, j, k, 6]) * 0.5f0
    @inbounds μi =  1.458f-6*T*sqrt(T)/(T+110.4f0)
    @inbounds λi =  1004.5f0*μi/0.7f0

    @inbounds ∂u∂ξ = 0.5f0*(c23*(Q[i+1, j, k, 2] + Q[i+1, j, k-1, 2] - Q[i-1, j, k, 2] - Q[i-1, j, k-1, 2]) -
                          c12*(Q[i+2, j, k, 2] + Q[i+2, j, k-1, 2] - Q[i-2, j, k, 2] - Q[i-2, j, k-1, 2]))
    @inbounds ∂v∂ξ = 0.5f0*(c23*(Q[i+1, j, k, 3] + Q[i+1, j, k-1, 3] - Q[i-1, j, k, 3] - Q[i-1, j, k-1, 3]) -
                          c12*(Q[i+2, j, k, 3] + Q[i+2, j, k-1, 3] - Q[i-2, j, k, 3] - Q[i-2, j, k-1, 3]))
    @inbounds ∂w∂ξ = 0.5f0*(c23*(Q[i+1, j, k, 4] + Q[i+1, j, k-1, 4] - Q[i-1, j, k, 4] - Q[i-1, j, k-1, 4]) -
                          c12*(Q[i+2, j, k, 4] + Q[i+2, j, k-1, 4] - Q[i-2, j, k, 4] - Q[i-2, j, k-1, 4]))
    @inbounds ∂T∂ξ = 0.5f0*(c23*(Q[i+1, j, k, 6] + Q[i+1, j, k-1, 6] - Q[i-1, j, k, 6] - Q[i-1, j, k-1, 6]) -
                          c12*(Q[i+2, j, k, 6] + Q[i+2, j, k-1, 6] - Q[i-2, j, k, 6] - Q[i-2, j, k-1, 6]))

    @inbounds ∂u∂η = 0.5f0*(c23*(Q[i, j+1, k, 2] + Q[i, j+1, k-1, 2] - Q[i, j-1, k, 2] - Q[i, j-1, k-1, 2]) -
                          c12*(Q[i, j+2, k, 2] + Q[i, j+2, k-1, 2] - Q[i, j-2, k, 2] - Q[i, j-2, k-1, 2]))
    @inbounds ∂v∂η = 0.5f0*(c23*(Q[i, j+1, k, 3] + Q[i, j+1, k-1, 3] - Q[i, j-1, k, 3] - Q[i, j-1, k-1, 3]) -
                          c12*(Q[i, j+2, k, 3] + Q[i, j+2, k-1, 3] - Q[i, j-2, k, 3] - Q[i, j-2, k-1, 3]))
    @inbounds ∂w∂η = 0.5f0*(c23*(Q[i, j+1, k, 4] + Q[i, j+1, k-1, 4] - Q[i, j-1, k, 4] - Q[i, j-1, k-1, 4]) -
                          c12*(Q[i, j+2, k, 4] + Q[i, j+2, k-1, 4] - Q[i, j-2, k, 4] - Q[i, j-2, k-1, 4]))
    @inbounds ∂T∂η = 0.5f0*(c23*(Q[i, j+1, k, 6] + Q[i, j+1, k-1, 6] - Q[i, j-1, k, 6] - Q[i, j-1, k-1, 6]) -
                          c12*(Q[i, j+2, k, 6] + Q[i, j+2, k-1, 6] - Q[i, j-2, k, 6] - Q[i, j-2, k-1, 6]))

    @inbounds ∂u∂ζ = 1.25f0*(Q[i, j, k, 2] - Q[i, j, k-1, 2]) - c12*(Q[i, j, k+1, 2] - Q[i, j, k-2, 2])
    @inbounds ∂v∂ζ = 1.25f0*(Q[i, j, k, 3] - Q[i, j, k-1, 3]) - c12*(Q[i, j, k+1, 3] - Q[i, j, k-2, 3])
    @inbounds ∂w∂ζ = 1.25f0*(Q[i, j, k, 4] - Q[i, j, k-1, 4]) - c12*(Q[i, j, k+1, 4] - Q[i, j, k-2, 4])
    @inbounds ∂T∂ζ = 1.25f0*(Q[i, j, k, 6] - Q[i, j, k-1, 6]) - c12*(Q[i, j, k+1, 6] - Q[i, j, k-2, 6])

    @inbounds u = (Q[i, j, k-1, 2] + Q[i, j, k, 2]) * 0.5f0
    @inbounds v = (Q[i, j, k-1, 3] + Q[i, j, k, 3]) * 0.5f0
    @inbounds w = (Q[i, j, k-1, 4] + Q[i, j, k, 4]) * 0.5f0

    dudx = (∂u∂ξ * ∂ξ∂x + ∂u∂η * ∂η∂x + ∂u∂ζ * ∂ζ∂x) * Jac
    dudy = (∂u∂ξ * ∂ξ∂y + ∂u∂η * ∂η∂y + ∂u∂ζ * ∂ζ∂y) * Jac
    dudz = (∂u∂ξ * ∂ξ∂z + ∂u∂η * ∂η∂z + ∂u∂ζ * ∂ζ∂z) * Jac

    dvdx = (∂v∂ξ * ∂ξ∂x + ∂v∂η * ∂η∂x + ∂v∂ζ * ∂ζ∂x) * Jac
    dvdy = (∂v∂ξ * ∂ξ∂y + ∂v∂η * ∂η∂y + ∂v∂ζ * ∂ζ∂y) * Jac
    dvdz = (∂v∂ξ * ∂ξ∂z + ∂v∂η * ∂η∂z + ∂v∂ζ * ∂ζ∂z) * Jac

    dwdx = (∂w∂ξ * ∂ξ∂x + ∂w∂η * ∂η∂x + ∂w∂ζ * ∂ζ∂x) * Jac
    dwdy = (∂w∂ξ * ∂ξ∂y + ∂w∂η * ∂η∂y + ∂w∂ζ * ∂ζ∂y) * Jac
    dwdz = (∂w∂ξ * ∂ξ∂z + ∂w∂η * ∂η∂z + ∂w∂ζ * ∂ζ∂z) * Jac

    dTdx = (∂T∂ξ * ∂ξ∂x + ∂T∂η * ∂η∂x + ∂T∂ζ * ∂ζ∂x) * Jac
    dTdy = (∂T∂ξ * ∂ξ∂y + ∂T∂η * ∂η∂y + ∂T∂ζ * ∂ζ∂y) * Jac
    dTdz = (∂T∂ξ * ∂ξ∂z + ∂T∂η * ∂η∂z + ∂T∂ζ * ∂ζ∂z) * Jac

    div = dudx + dvdy + dwdz

    if LES_smag
        Cs = 0.1f0
        Prt = 0.9f0
        @inbounds ρ = 0.5f0 * (Q[i, j, k, 1] + Q[i, j, k-1, 1])

        @fastmath Sijmag = sqrt(2*(dudx^2 + dvdy^2 + dwdz^2 + 
                               2*((0.5f0*(dudy+dvdx))^2 + (0.5f0*(dudz+dwdx))^2 +(0.5f0*(dvdz+dwdy))^2))) # √2|sij|
      
        @fastmath μt = ρ * (Cs/Jac^(1/3f0))^2 * Sijmag #ρ(csΔ)^2 * Sijmag
      
        λt = 1004.5f0 * μt / Prt # cp = Rg*γ/(γ-1)
        μratio_z[i-NG, j-NG, k-NG] = μt/μi

        μi += μt
        λi += λt
    elseif LES_wale
        Cw = 0.325f0
        Prt = 0.9f0
        @inbounds ρ = 0.5f0 * (Q[i, j, k, 1] + Q[i, j, k-1, 1])

        @fastmath S = sqrt(dudx^2 + dvdy^2 + dwdz^2 + 
        2*((0.5f0*(dudy+dvdx))^2 + (0.5f0*(dudz+dwdx))^2 +(0.5f0*(dvdz+dwdy))^2))

        Sd11 = dudx*dudx + dudy*dvdx + dudz*dwdx
        Sd22 = dvdx*dudy + dvdy*dvdy + dvdz*dwdy
        Sd33 = dwdx*dudz + dwdy*dvdz + dwdz*dwdz
        trSd = 1/3f0*(Sd11 + Sd22 + Sd33)
        Sd11 -= trSd
        Sd22 -= trSd
        Sd33 -= trSd
        Sd12 = 0.5f0*(dudx*dvdx + dvdx*dvdy + dwdx*dvdz + dudy*dudx + dvdy*dudy + dwdy*dudz)
        Sd13 = 0.5f0*(dudx*dwdx + dvdx*dwdy + dwdx*dwdz + dudz*dudx + dvdz*dudy + dwdz*dudz)
        Sd23 = 0.5f0*(dudy*dwdx + dvdy*dwdy + dwdy*dwdz + dudz*dvdx + dvdz*dvdy + dwdz*dvdz)
        @fastmath Sd = sqrt(Sd11^2 + Sd22^2 + Sd33^2 + 2 * (Sd12^2 + Sd13^2 + Sd23^2))
        @fastmath D = Sd^3/(S^5 + Sd^2.5f0)
        @fastmath μt = min(ρ * (Cw/Jac^(1/3f0))^2 * D, 6*μi)
      
        λt = 1004.5f0 * μt / Prt # cp = Rg*γ/(γ-1)
        μratio_z[i-NG, j-NG, k-NG] = μt/μi

        μi += μt
        λi += λt
    end

    τ11 = μi*(2*dudx - c23*div)
    τ12 = μi*(dudy + dvdx)
    τ13 = μi*(dudz + dwdx)
    τ22 = μi*(2*dvdy - c23*div)
    τ23 = μi*(dwdy + dvdz)
    τ33 = μi*(2*dwdz - c23*div)

    @inbounds E1 = u * τ11 + v * τ12 + w * τ13 + λi * dTdx
    @inbounds E2 = u * τ12 + v * τ22 + w * τ23 + λi * dTdy
    @inbounds E3 = u * τ13 + v * τ23 + w * τ33 + λi * dTdz

    @inbounds Fv_z[i-NG, j-NG, k-NG, 1] = ∂ζ∂x * τ11 + ∂ζ∂y * τ12 + ∂ζ∂z * τ13
    @inbounds Fv_z[i-NG, j-NG, k-NG, 2] = ∂ζ∂x * τ12 + ∂ζ∂y * τ22 + ∂ζ∂z * τ23
    @inbounds Fv_z[i-NG, j-NG, k-NG, 3] = ∂ζ∂x * τ13 + ∂ζ∂y * τ23 + ∂ζ∂z * τ33
    @inbounds Fv_z[i-NG, j-NG, k-NG, 4] = ∂ζ∂x * E1 + ∂ζ∂y * E2 + ∂ζ∂z * E3
    return
end
