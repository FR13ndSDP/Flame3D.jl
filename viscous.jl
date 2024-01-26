# #Range: 3 -> N+2*NG-2
function viscousFlux(Fv_x, Fv_y, Fv_z, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, λ, μ, Fh, consts)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG-2 || j > Ny+2*NG-2 || k > Nz+2*NG-2 || i < 3 || j < 3 || k < 3
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
    @inbounds μi = μ[i, j, k]
    @inbounds λi = λ[i, j, k]
    # C_s = consts.C_s
    # T_s = consts.T_s
    # Pr = consts.Pr
    # Cp = consts.Cp
    # μi = C_s * T * CUDA.sqrt(T)/(T + T_s)
    # λi = Cp*μi/Pr

    c1::Float64 = consts.CD4[1]
    c2::Float64 = consts.CD4[2]

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

    τ11 = μi*(2*dudx - c2*div)
    τ12 = μi*(dudy + dvdx)
    τ13 = μi*(dudz + dwdx)
    τ22 = μi*(2*dvdy - c2*div)
    τ23 = μi*(dwdy + dvdz)
    τ33 = μi*(2*dwdz - c2*div)

    @inbounds E1 = u * τ11 + v * τ12 + w * τ13 + λi * dTdx + Fh[i, j, k, 1]
    @inbounds E2 = u * τ12 + v * τ22 + w * τ23 + λi * dTdy + Fh[i, j, k, 2]
    @inbounds E3 = u * τ13 + v * τ23 + w * τ33 + λi * dTdz + Fh[i, j, k, 3]

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

# #Range: 3 -> N+2*NG-2
function specViscousFlux(Fv_x, Fv_y, Fv_z, Q, Yi, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, D, Fh, thermo, consts)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG-2 || j > Ny+2*NG-2 || k > Nz+2*NG-2 || i < 3 || j < 3 || k < 3
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
    @inbounds ρ = Q[i, j, k, 1]
    @inbounds T = Q[i, j, k, 6]
    T2::Float64 = T * T
    T3::Float64 = T2 * T
    T4::Float64 = T2 * T2
    T5::Float64 = T3 * T2

    @inbounds Fh[i-2, j-2, k-2, 1] = 0
    @inbounds Fh[i-2, j-2, k-2, 2] = 0
    @inbounds Fh[i-2, j-2, k-2, 3] = 0

    c1::Float64 = consts.CD4[1]
    c2::Float64 = consts.CD4[2]

    for n = 1:Nspecs
        @inbounds Di = D[i, j, k, n]
        @inbounds ∂Y∂ξ = c1*(Yi[i-2, j, k, n] - Yi[i+2, j, k, n]) + c2*(Yi[i-1, j, k, n] - Yi[i+1, j, k, n])
        @inbounds ∂Y∂η = c1*(Yi[i, j-2, k, n] - Yi[i, j+2, k, n]) + c2*(Yi[i, j-1, k, n] - Yi[i, j+1, k, n])
        @inbounds ∂Y∂ζ = c1*(Yi[i, j, k-2, n] - Yi[i, j, k+2, n]) + c2*(Yi[i, j, k-1, n] - Yi[i, j, k+1, n])

        dYdx = (∂Y∂ξ * ∂ξ∂x + ∂Y∂η * ∂η∂x + ∂Y∂ζ * ∂ζ∂x) * Jac
        dYdy = (∂Y∂ξ * ∂ξ∂y + ∂Y∂η * ∂η∂y + ∂Y∂ζ * ∂ζ∂y) * Jac
        dYdz = (∂Y∂ξ * ∂ξ∂z + ∂Y∂η * ∂η∂z + ∂Y∂ζ * ∂ζ∂z) * Jac

        tmp1 = ρ*Di*dYdx
        tmp2 = ρ*Di*dYdy
        tmp3 = ρ*Di*dYdz

        @inbounds Fv_x[i-2, j-2, k-2, n] = tmp1 * ∂ξ∂x + tmp2 * ∂ξ∂y + tmp3 * ∂ξ∂z
        @inbounds Fv_y[i-2, j-2, k-2, n] = tmp2 * ∂η∂x + tmp2 * ∂η∂y + tmp3 * ∂η∂z
        @inbounds Fv_z[i-2, j-2, k-2, n] = tmp3 * ∂ζ∂x + tmp2 * ∂ζ∂y + tmp3 * ∂ζ∂z

        h = h_specs(T, T2, T3, T4, T5, n, thermo)
        @inbounds Fh[i-2, j-2, k-2, 1] += tmp1 * h
        @inbounds Fh[i-2, j-2, k-2, 2] += tmp2 * h
        @inbounds Fh[i-2, j-2, k-2, 3] += tmp3 * h
    end
    return
end
