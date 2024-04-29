function viscousFlux(Fv_x, Fv_y, Fv_z, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG-2 || j > Nyp+2*NG-2 || k > Nzp+2*NG-2 || i < 3 || j < 3 || k < 3
        return
    end

    @inbounds ∂ξ∂x = dξdx[i, j, k]
    @inbounds ∂ξ∂y = dξdy[i, j, k]
    @inbounds ∂ξ∂z = dξdz[i, j, k]
    @inbounds ∂η∂x = dηdx[i, j, k]
    @inbounds ∂η∂y = dηdy[i, j, k]
    @inbounds ∂η∂z = dηdz[i, j, k]
    @inbounds ∂ζ∂x = dζdx[i, j, k]
    @inbounds ∂ζ∂y = dζdy[i, j, k]
    @inbounds ∂ζ∂z = dζdz[i, j, k]

    @inbounds Jac = J[i, j, k]
    @inbounds T = Q[i, j, k, 6]
    μi::Float32 =  C_s*T*sqrt(T)/(T+T_s)
    λi::Float32 =  Cp*μi/Pr

    c1::Float32 = 1/12f0
    c2::Float32 = -2/3f0
    c23::Float32 = 2/3f0

    @inbounds ∂u∂ξ = c1*(Q[i-2, j, k, 2] - Q[i+2, j, k, 2]) + c2*(Q[i-1, j, k, 2] - Q[i+1, j, k, 2])
    @inbounds ∂v∂ξ = c1*(Q[i-2, j, k, 3] - Q[i+2, j, k, 3]) + c2*(Q[i-1, j, k, 3] - Q[i+1, j, k, 3])
    @inbounds ∂w∂ξ = c1*(Q[i-2, j, k, 4] - Q[i+2, j, k, 4]) + c2*(Q[i-1, j, k, 4] - Q[i+1, j, k, 4])
    @inbounds ∂T∂ξ = c1*(Q[i-2, j, k, 6] - Q[i+2, j, k, 6]) + c2*(Q[i-1, j, k, 6] - Q[i+1, j, k, 6])

    @inbounds ∂u∂η = c1*(Q[i, j-2, k, 2] - Q[i, j+2, k, 2]) + c2*(Q[i, j-1, k, 2] - Q[i, j+1, k, 2])
    @inbounds ∂v∂η = c1*(Q[i, j-2, k, 3] - Q[i, j+2, k, 3]) + c2*(Q[i, j-1, k, 3] - Q[i, j+1, k, 3])
    @inbounds ∂w∂η = c1*(Q[i, j-2, k, 4] - Q[i, j+2, k, 4]) + c2*(Q[i, j-1, k, 4] - Q[i, j+1, k, 4])
    @inbounds ∂T∂η = c1*(Q[i, j-2, k, 6] - Q[i, j+2, k, 6]) + c2*(Q[i, j-1, k, 6] - Q[i, j+1, k, 6])

    @inbounds ∂u∂ζ = c1*(Q[i, j, k-2, 2] - Q[i, j, k+2, 2]) + c2*(Q[i, j, k-1, 2] - Q[i, j, k+1, 2])
    @inbounds ∂v∂ζ = c1*(Q[i, j, k-2, 3] - Q[i, j, k+2, 3]) + c2*(Q[i, j, k-1, 3] - Q[i, j, k+1, 3])
    @inbounds ∂w∂ζ = c1*(Q[i, j, k-2, 4] - Q[i, j, k+2, 4]) + c2*(Q[i, j, k-1, 4] - Q[i, j, k+1, 4])
    @inbounds ∂T∂ζ = c1*(Q[i, j, k-2, 6] - Q[i, j, k+2, 6]) + c2*(Q[i, j, k-1, 6] - Q[i, j, k+1, 6])

    @inbounds u = Q[i, j, k, 2]
    @inbounds v = Q[i, j, k, 3]
    @inbounds w = Q[i, j, k, 4]

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
        @inbounds ρ = Q[i, j, k, 1]

        @fastmath Sijmag = sqrt(2*(dudx^2 + dvdy^2 + dwdz^2 + 
                               2*((0.5f0*(dudy+dvdx))^2 + (0.5f0*(dudz+dwdx))^2 +(0.5f0*(dvdz+dwdy))^2))) # √2|sij|
      
        @fastmath μt = min(ρ * (Cs/Jac^(1/3f0))^2 * Sijmag, 2*μi) #ρ(csΔ)^2 * Sijmag

        λt = Cp * μt / Prt # cp = Rg*γ/(γ-1)

        μi += μt
        λi += λt
    elseif LES_wale
        Cw = 0.325f0
        Prt = 0.9f0
        @inbounds ρ = Q[i, j, k, 1]

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
        @fastmath μt = min(ρ * (Cw/Jac^(1/3f0))^2 * D, 2*μi)
      
        λt = Cp * μt / Prt # cp = Rg*γ/(γ-1)

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

    @inbounds Fv_x[i-2, j-2, k-2, 1] = ∂ξ∂x * τ11 + ∂ξ∂y * τ12 + ∂ξ∂z * τ13
    @inbounds Fv_x[i-2, j-2, k-2, 2] = ∂ξ∂x * τ12 + ∂ξ∂y * τ22 + ∂ξ∂z * τ23
    @inbounds Fv_x[i-2, j-2, k-2, 3] = ∂ξ∂x * τ13 + ∂ξ∂y * τ23 + ∂ξ∂z * τ33
    @inbounds Fv_x[i-2, j-2, k-2, 4] = ∂ξ∂x * E1 + ∂ξ∂y * E2 + ∂ξ∂z * E3

    @inbounds Fv_y[i-2, j-2, k-2, 1] = ∂η∂x * τ11 + ∂η∂y * τ12 + ∂η∂z * τ13
    @inbounds Fv_y[i-2, j-2, k-2, 2] = ∂η∂x * τ12 + ∂η∂y * τ22 + ∂η∂z * τ23
    @inbounds Fv_y[i-2, j-2, k-2, 3] = ∂η∂x * τ13 + ∂η∂y * τ23 + ∂η∂z * τ33
    @inbounds Fv_y[i-2, j-2, k-2, 4] = ∂η∂x * E1 + ∂η∂y * E2 + ∂η∂z * E3

    @inbounds Fv_z[i-2, j-2, k-2, 1] = ∂ζ∂x * τ11 + ∂ζ∂y * τ12 + ∂ζ∂z * τ13
    @inbounds Fv_z[i-2, j-2, k-2, 2] = ∂ζ∂x * τ12 + ∂ζ∂y * τ22 + ∂ζ∂z * τ23
    @inbounds Fv_z[i-2, j-2, k-2, 3] = ∂ζ∂x * τ13 + ∂ζ∂y * τ23 + ∂ζ∂z * τ33
    @inbounds Fv_z[i-2, j-2, k-2, 4] = ∂ζ∂x * E1 + ∂ζ∂y * E2 + ∂ζ∂z * E3
    return
end