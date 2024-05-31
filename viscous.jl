function viscousFlux_x(Fv_x, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, λ, μ, Fh)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

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
    @inbounds μi =  (μ[i-1, j, k] + μ[i, j, k]) * 0.5f0 
    @inbounds λi =  (λ[i-1, j, k] + λ[i, j, k]) * 0.5f0

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

    τ11 = μi*(2*dudx - c23*div)
    τ12 = μi*(dudy + dvdx)
    τ13 = μi*(dudz + dwdx)
    τ22 = μi*(2*dvdy - c23*div)
    τ23 = μi*(dwdy + dvdz)
    τ33 = μi*(2*dwdz - c23*div)

    @inbounds E1 = u * τ11 + v * τ12 + w * τ13 + λi * dTdx + Fh[i-NG, j-NG, k-NG, 1]
    @inbounds E2 = u * τ12 + v * τ22 + w * τ23 + λi * dTdy + Fh[i-NG, j-NG, k-NG, 2]
    @inbounds E3 = u * τ13 + v * τ23 + w * τ33 + λi * dTdz + Fh[i-NG, j-NG, k-NG, 3]

    @inbounds Fv_x[i-NG, j-NG, k-NG, 1] = ∂ξ∂x * τ11 + ∂ξ∂y * τ12 + ∂ξ∂z * τ13
    @inbounds Fv_x[i-NG, j-NG, k-NG, 2] = ∂ξ∂x * τ12 + ∂ξ∂y * τ22 + ∂ξ∂z * τ23
    @inbounds Fv_x[i-NG, j-NG, k-NG, 3] = ∂ξ∂x * τ13 + ∂ξ∂y * τ23 + ∂ξ∂z * τ33
    @inbounds Fv_x[i-NG, j-NG, k-NG, 4] = ∂ξ∂x * E1 + ∂ξ∂y * E2 + ∂ξ∂z * E3
    return
end

function viscousFlux_y(Fv_y, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, λ, μ, Fh)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

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
    @inbounds μi =  (μ[i, j-1, k] + μ[i, j, k]) * 0.5f0 
    @inbounds λi =  (λ[i, j-1, k] + λ[i, j, k]) * 0.5f0

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

    τ11 = μi*(2*dudx - c23*div)
    τ12 = μi*(dudy + dvdx)
    τ13 = μi*(dudz + dwdx)
    τ22 = μi*(2*dvdy - c23*div)
    τ23 = μi*(dwdy + dvdz)
    τ33 = μi*(2*dwdz - c23*div)

    @inbounds E1 = u * τ11 + v * τ12 + w * τ13 + λi * dTdx + Fh[i-NG, j-NG, k-NG, 1]
    @inbounds E2 = u * τ12 + v * τ22 + w * τ23 + λi * dTdy + Fh[i-NG, j-NG, k-NG, 2]
    @inbounds E3 = u * τ13 + v * τ23 + w * τ33 + λi * dTdz + Fh[i-NG, j-NG, k-NG, 3]

    @inbounds Fv_y[i-NG, j-NG, k-NG, 1] = ∂η∂x * τ11 + ∂η∂y * τ12 + ∂η∂z * τ13
    @inbounds Fv_y[i-NG, j-NG, k-NG, 2] = ∂η∂x * τ12 + ∂η∂y * τ22 + ∂η∂z * τ23
    @inbounds Fv_y[i-NG, j-NG, k-NG, 3] = ∂η∂x * τ13 + ∂η∂y * τ23 + ∂η∂z * τ33
    @inbounds Fv_y[i-NG, j-NG, k-NG, 4] = ∂η∂x * E1 + ∂η∂y * E2 + ∂η∂z * E3
    return
end

function viscousFlux_z(Fv_z, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, λ, μ, Fh)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

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
    @inbounds μi =  (μ[i, j, k-1] + μ[i, j, k]) * 0.5f0 
    @inbounds λi =  (λ[i, j, k-1] + λ[i, j, k]) * 0.5f0

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

    τ11 = μi*(2*dudx - c23*div)
    τ12 = μi*(dudy + dvdx)
    τ13 = μi*(dudz + dwdx)
    τ22 = μi*(2*dvdy - c23*div)
    τ23 = μi*(dwdy + dvdz)
    τ33 = μi*(2*dwdz - c23*div)

    @inbounds E1 = u * τ11 + v * τ12 + w * τ13 + λi * dTdx + Fh[i-NG, j-NG, k-NG, 1]
    @inbounds E2 = u * τ12 + v * τ22 + w * τ23 + λi * dTdy + Fh[i-NG, j-NG, k-NG, 2]
    @inbounds E3 = u * τ13 + v * τ23 + w * τ33 + λi * dTdz + Fh[i-NG, j-NG, k-NG, 3]

    @inbounds Fv_z[i-NG, j-NG, k-NG, 1] = ∂ζ∂x * τ11 + ∂ζ∂y * τ12 + ∂ζ∂z * τ13
    @inbounds Fv_z[i-NG, j-NG, k-NG, 2] = ∂ζ∂x * τ12 + ∂ζ∂y * τ22 + ∂ζ∂z * τ23
    @inbounds Fv_z[i-NG, j-NG, k-NG, 3] = ∂ζ∂x * τ13 + ∂ζ∂y * τ23 + ∂ζ∂z * τ33
    @inbounds Fv_z[i-NG, j-NG, k-NG, 4] = ∂ζ∂x * E1 + ∂ζ∂y * E2 + ∂ζ∂z * E3
    return
end

function specViscousFlux_x(Fv_x, Q, Yi, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, D, Fh, thermo)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+NG+1 || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

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
    @inbounds ρ = (Q[i-1, j, k, 1] + Q[i, j, k, 1]) * 0.5f0
    @inbounds T = (Q[i-1, j, k, 6] + Q[i, j, k, 6]) * 0.5f0

    # diffusion velocity
    Vk1 = MVector{Nspecs, Float32}(undef)
    Vk2 = MVector{Nspecs, Float32}(undef)
    Vk3 = MVector{Nspecs, Float32}(undef)
    hi = MVector{Nspecs, Float32}(undef)
    h_specs(hi, T, thermo)
    @inbounds Fh[i-NG, j-NG, k-NG, 1] = 0
    @inbounds Fh[i-NG, j-NG, k-NG, 2] = 0
    @inbounds Fh[i-NG, j-NG, k-NG, 3] = 0

    sum1::Float32 = 0
    sum2::Float32 = 0
    sum3::Float32 = 0
    for n = 1:Nspecs
        @inbounds ρDi = (D[i-1, j, k, n] + D[i, j, k, n]) * ρ
        @inbounds ∂Y∂ξ = 1.25f0*(Yi[i, j, k, n] - Yi[i-1, j, k, n]) -
                        c12*(Yi[i+1, j, k, n] - Yi[i-2, j, k, n])
        @inbounds ∂Y∂η = 0.25f0 * (Yi[i, j+1, k, n] + Yi[i-1, j+1, k, n] - 
                                 Yi[i, j-1, k, n] - Yi[i-1, j-1, k, n])
        @inbounds ∂Y∂ζ = 0.25f0 * (Yi[i, j, k+1, n] + Yi[i-1, j, k+1, n] -
                                 Yi[i, j, k-1, n] - Yi[i-1, j, k-1, n])

        Vx = (∂Y∂ξ * ∂ξ∂x + ∂Y∂η * ∂η∂x + ∂Y∂ζ * ∂ζ∂x) * Jac * ρDi
        Vy = (∂Y∂ξ * ∂ξ∂y + ∂Y∂η * ∂η∂y + ∂Y∂ζ * ∂ζ∂y) * Jac * ρDi
        Vz = (∂Y∂ξ * ∂ξ∂z + ∂Y∂η * ∂η∂z + ∂Y∂ζ * ∂ζ∂z) * Jac * ρDi

        @inbounds Vk1[n] = Vx
        @inbounds Vk2[n] = Vy
        @inbounds Vk3[n] = Vz

        sum1 += Vx
        sum2 += Vy
        sum3 += Vz
    end

    for n = 1:Nspecs
        @inbounds Yn = 0.5f0 * (Yi[i-1, j, k, n] + Yi[i, j, k, n])
        @inbounds hn = hi[n]
        @inbounds V1 = Vk1[n] - sum1 * Yn
        @inbounds V2 = Vk2[n] - sum2 * Yn
        @inbounds V3 = Vk3[n] - sum3 * Yn

        @inbounds Fv_x[i-NG, j-NG, k-NG, n] = V1 * ∂ξ∂x + V2 * ∂ξ∂y + V3 * ∂ξ∂z
        @inbounds Fh[i-NG, j-NG, k-NG, 1] += V1 * hn
        @inbounds Fh[i-NG, j-NG, k-NG, 2] += V2 * hn
        @inbounds Fh[i-NG, j-NG, k-NG, 3] += V3 * hn
    end
    return
end

function specViscousFlux_y(Fv_y, Q, Yi, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, D, Fh, thermo)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+NG || j > Nyp+NG+1 || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

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
    @inbounds ρ = (Q[i, j-1, k, 1] + Q[i, j, k, 1]) * 0.5f0
    @inbounds T = (Q[i, j-1, k, 6] + Q[i, j, k, 6]) * 0.5f0

    # diffusion velocity
    Vk1 = MVector{Nspecs, Float32}(undef)
    Vk2 = MVector{Nspecs, Float32}(undef)
    Vk3 = MVector{Nspecs, Float32}(undef)
    hi = MVector{Nspecs, Float32}(undef)
    h_specs(hi, T, thermo)
    @inbounds Fh[i-NG, j-NG, k-NG, 1] = 0
    @inbounds Fh[i-NG, j-NG, k-NG, 2] = 0
    @inbounds Fh[i-NG, j-NG, k-NG, 3] = 0

    sum1::Float32 = 0
    sum2::Float32 = 0
    sum3::Float32 = 0
    for n = 1:Nspecs
        @inbounds ρDi = (D[i, j-1, k, n] + D[i, j, k, n]) * ρ
        @inbounds ∂Y∂ξ = 0.25f0 * (Yi[i+1, j, k, n] + Yi[i+1, j-1, k, n] - 
                                 Yi[i-1, j, k, n] - Yi[i-1, j-1, k, n])
        @inbounds ∂Y∂η = 1.25f0*(Yi[i, j, k, n] - Yi[i, j-1, k, n]) - 
                         c12*(Yi[i, j+1, k, n] - Yi[i, j-2, k, n])
        @inbounds ∂Y∂ζ = 0.25f0 * (Yi[i, j, k+1, n] + Yi[i, j-1, k+1, n] -
                                 Yi[i, j, k-1, n] - Yi[i, j-1, k-1, n])

        Vx = (∂Y∂ξ * ∂ξ∂x + ∂Y∂η * ∂η∂x + ∂Y∂ζ * ∂ζ∂x) * Jac * ρDi
        Vy = (∂Y∂ξ * ∂ξ∂y + ∂Y∂η * ∂η∂y + ∂Y∂ζ * ∂ζ∂y) * Jac * ρDi
        Vz = (∂Y∂ξ * ∂ξ∂z + ∂Y∂η * ∂η∂z + ∂Y∂ζ * ∂ζ∂z) * Jac * ρDi

        @inbounds Vk1[n] = Vx
        @inbounds Vk2[n] = Vy
        @inbounds Vk3[n] = Vz

        sum1 += Vx
        sum2 += Vy
        sum3 += Vz
    end

    for n = 1:Nspecs
        @inbounds Yn = 0.5f0 * (Yi[i, j-1, k, n] + Yi[i, j, k, n])
        @inbounds hn = hi[n]
        @inbounds V1 = Vk1[n] - sum1 * Yn
        @inbounds V2 = Vk2[n] - sum2 * Yn
        @inbounds V3 = Vk3[n] - sum3 * Yn

        @inbounds Fv_y[i-NG, j-NG, k-NG, n] = V1 * ∂η∂x + V2 * ∂η∂y + V3 * ∂η∂z
        @inbounds Fh[i-NG, j-NG, k-NG, 1] += V1 * hn
        @inbounds Fh[i-NG, j-NG, k-NG, 2] += V2 * hn
        @inbounds Fh[i-NG, j-NG, k-NG, 3] += V3 * hn
    end
    return
end

function specViscousFlux_z(Fv_z, Q, Yi, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, D, Fh, thermo)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG+1 || i < NG+1 || j < NG+1 || k < NG+1
        return
    end
    
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
    @inbounds ρ = (Q[i, j, k-1, 1] + Q[i, j, k, 1]) * 0.5f0
    @inbounds T = (Q[i, j, k-1, 6] + Q[i, j, k, 6]) * 0.5f0

    # diffusion velocity
    Vk1 = MVector{Nspecs, Float32}(undef)
    Vk2 = MVector{Nspecs, Float32}(undef)
    Vk3 = MVector{Nspecs, Float32}(undef)
    hi = MVector{Nspecs, Float32}(undef)
    h_specs(hi, T, thermo)
    @inbounds Fh[i-NG, j-NG, k-NG, 1] = 0
    @inbounds Fh[i-NG, j-NG, k-NG, 2] = 0
    @inbounds Fh[i-NG, j-NG, k-NG, 3] = 0

    sum1::Float32 = 0
    sum2::Float32 = 0
    sum3::Float32 = 0
    for n = 1:Nspecs
        @inbounds ρDi = (D[i, j, k-1, n] + D[i, j, k, n]) * ρ
        @inbounds ∂Y∂ξ = 0.25f0 * (Yi[i+1, j, k, n] + Yi[i+1, j, k-1, n] - 
                                 Yi[i-1, j, k, n] - Yi[i-1, j, k-1, n])
        @inbounds ∂Y∂η = 0.25f0 * (Yi[i, j+1, k, n] + Yi[i, j+1, k-1, n] - 
                                 Yi[i, j-1, k, n] - Yi[i, j-1, k-1, n])
        @inbounds ∂Y∂ζ = 1.25f0*(Yi[i, j, k, n] - Yi[i, j, k-1, n]) - 
                         c12*(Yi[i, j, k+1, n] - Yi[i, j, k-2, n])

        Vx = (∂Y∂ξ * ∂ξ∂x + ∂Y∂η * ∂η∂x + ∂Y∂ζ * ∂ζ∂x) * Jac * ρDi
        Vy = (∂Y∂ξ * ∂ξ∂y + ∂Y∂η * ∂η∂y + ∂Y∂ζ * ∂ζ∂y) * Jac * ρDi
        Vz = (∂Y∂ξ * ∂ξ∂z + ∂Y∂η * ∂η∂z + ∂Y∂ζ * ∂ζ∂z) * Jac * ρDi

        @inbounds Vk1[n] = Vx
        @inbounds Vk2[n] = Vy
        @inbounds Vk3[n] = Vz

        sum1 += Vx
        sum2 += Vy
        sum3 += Vz
    end

    for n = 1:Nspecs
        @inbounds Yn = 0.5f0 * (Yi[i, j, k-1, n] + Yi[i, j, k, n])
        @inbounds hn = hi[n]
        @inbounds V1 = Vk1[n] - sum1 * Yn
        @inbounds V2 = Vk2[n] - sum2 * Yn
        @inbounds V3 = Vk3[n] - sum3 * Yn

        @inbounds Fv_z[i-NG, j-NG, k-NG, n] = V1 * ∂ζ∂x + V2 * ∂ζ∂y + V3 * ∂ζ∂z
        @inbounds Fh[i-NG, j-NG, k-NG, 1] += V1 * hn
        @inbounds Fh[i-NG, j-NG, k-NG, 2] += V2 * hn
        @inbounds Fh[i-NG, j-NG, k-NG, 3] += V3 * hn
    end
    return
end