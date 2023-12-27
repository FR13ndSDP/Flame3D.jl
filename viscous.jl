# #Range: 3 -> N+2*NG-2
function viscousFlux(Fv_x, Fv_y, Fv_z, Q, NG, Nx, Ny, Nz, Pr, Cp, C_s, T_s, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nx+2*NG-2 || j > Ny+2*NG-2 || k > Nz+2*NG-2 || i < 3 || j < 3 || k < 3
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
    μ = C_s * T * CUDA.sqrt(T)/(T + T_s)
    κ = Cp*μ/Pr

    c1::Float64 = 1/12
    c2::Float64 = 2/3
    @inbounds ∂u∂ξ = c1*(Q[i-2, j, k, 2] - Q[i+2, j, k, 2]) - c2*(Q[i-1, j, k, 2] - Q[i+1, j, k, 2])
    @inbounds ∂u∂η = c1*(Q[i, j-2, k, 2] - Q[i, j+2, k, 2]) - c2*(Q[i, j-1, k, 2] - Q[i, j+1, k, 2])
    @inbounds ∂u∂ζ = c1*(Q[i, j, k-2, 2] - Q[i, j, k+2, 2]) - c2*(Q[i, j, k-1, 2] - Q[i, j, k+1, 2])

    @inbounds ∂v∂ξ = c1*(Q[i-2, j, k, 3] - Q[i+2, j, k, 3]) - c2*(Q[i-1, j, k, 3] - Q[i+1, j, k, 3])
    @inbounds ∂v∂η = c1*(Q[i, j-2, k, 3] - Q[i, j+2, k, 3]) - c2*(Q[i, j-1, k, 3] - Q[i, j+1, k, 3])
    @inbounds ∂v∂ζ = c1*(Q[i, j, k-2, 3] - Q[i, j, k+2, 3]) - c2*(Q[i, j, k-1, 3] - Q[i, j, k+1, 3])

    @inbounds ∂w∂ξ = c1*(Q[i-2, j, k, 4] - Q[i+2, j, k, 4]) - c2*(Q[i-1, j, k, 4] - Q[i+1, j, k, 4])
    @inbounds ∂w∂η = c1*(Q[i, j-2, k, 4] - Q[i, j+2, k, 4]) - c2*(Q[i, j-1, k, 4] - Q[i, j+1, k, 4])
    @inbounds ∂w∂ζ = c1*(Q[i, j, k-2, 4] - Q[i, j, k+2, 4]) - c2*(Q[i, j, k-1, 4] - Q[i, j, k+1, 4])

    @inbounds ∂T∂ξ = c1*(Q[i-2, j, k, 6] - Q[i+2, j, k, 6]) - c2*(Q[i-1, j, k, 6] - Q[i+1, j, k, 6])
    @inbounds ∂T∂η = c1*(Q[i, j-2, k, 6] - Q[i, j+2, k, 6]) - c2*(Q[i, j-1, k, 6] - Q[i, j+1, k, 6])
    @inbounds ∂T∂ζ = c1*(Q[i, j, k-2, 6] - Q[i, j, k+2, 6]) - c2*(Q[i, j, k-1, 6] - Q[i, j, k+1, 6])

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

    τ11 = μ*(2*dudx - c2*div)
    τ12 = μ*(dudy + dvdx)
    τ13 = μ*(dudz + dwdx)
    τ22 = μ*(2*dvdy - c2*div)
    τ23 = μ*(dwdy + dvdz)
    τ33 = μ*(2*dwdz - c2*div)

    E1 = u * τ11 + v * τ12 + w * τ13 + κ * dTdx
    E2 = u * τ12 + v * τ22 + w * τ23 + κ * dTdy
    E3 = u * τ13 + v * τ23 + w * τ33 + κ * dTdz

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
