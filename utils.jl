# Range: 1+NG -> N+NG
function c2Prim(U, Q)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Ny+NG || k > Nz+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    # correction
    @inbounds ρ = max(U[i, j, k, 1], CUDA.eps(Float64))
    @inbounds ρinv = 1/ρ 

    @inbounds u = U[i, j, k, 2]*ρinv # U
    @inbounds v = U[i, j, k, 3]*ρinv # V
    @inbounds w = U[i, j, k, 4]*ρinv # W
    @inbounds ei = max((U[i, j, k, 5] - 0.5*ρ*(u^2 + v^2 + w^2)), CUDA.eps(Float64))

    p::Float64 = 0.4 * ei
    T::Float64 = p/(ρ*287)

    @inbounds Q[i, j, k, 1] = ρ
    @inbounds Q[i, j, k, 2] = u
    @inbounds Q[i, j, k, 3] = v
    @inbounds Q[i, j, k, 4] = w
    @inbounds Q[i, j, k, 5] = p
    @inbounds Q[i, j, k, 6] = T
    @inbounds Q[i, j, k, 7] = ei
    return
end

function localstep(Q, dt, CFL, x, y, z)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Ny+NG || k > Nz+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    d1 = 0.5 * sqrt((x[i+1, j, k] - x[i-1, j, k])^2 + (y[i+1, j, k] - y[i-1, j, k])^2 + (z[i+1, j, k] - z[i-1, j, k])^2)
    d2 = 0.5 * sqrt((x[i, j+1, k] - x[i, j-1, k])^2 + (y[i, j+1, k] - y[i, j-1, k])^2 + (z[i, j+1, k] - z[i, j-1, k])^2)
    d3 = 0.5 * sqrt((x[i, j, k+1] - x[i, j, k-1])^2 + (y[i, j, k+1] - y[i, j, k-1])^2 + (z[i, j, k+1] - z[i, j, k-1])^2)
    minΔ = min(d1, d2, d3)

    u = abs(Q[i, j, k, 2])
    v = abs(Q[i, j, k, 3])
    w = abs(Q[i, j, k, 4])
    c = sqrt(1.4*Q[i, j, k, 5]/Q[i, j, k,1])
    maxu = max(u,v,w) + c
    dt[i-NG, j-NG, k-NG] = CFL * minΔ/maxu
    return
end

# Range: 1 -> N+2*NG
function prim2c(U, Q)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    @inbounds ρ = Q[i, j, k, 1]
    @inbounds u = Q[i, j, k, 2]
    @inbounds v = Q[i, j, k, 3]
    @inbounds w = Q[i, j, k, 4]
    @inbounds ei = Q[i, j, k, 7]

    @inbounds U[i, j, k, 1] = ρ
    @inbounds U[i, j, k, 2] = u * ρ
    @inbounds U[i, j, k, 3] = v * ρ
    @inbounds U[i, j, k, 4] = w * ρ
    @inbounds U[i, j, k, 5] = ei + 0.5 * ρ * (u^2 + v^2 + w^2)
    return
end

# Range: 1+NG -> N+NG
function linComb(U, Un, NV, a::Float64, b::Float64)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Ny+NG || k > Nz+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    for n = 1:NV
        @inbounds U[i, j, k, n] = U[i, j, k, n] * a + Un[i, j, k, n] * b
    end
    return
end
