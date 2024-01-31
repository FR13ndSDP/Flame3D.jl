# Range: 1+NG -> N+NG
function c2Prim(U, Q, ρi, thermo)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Ny+NG || k > Nz+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    # correction
    @inbounds ρ = max(U[i, j, k, 1], CUDA.eps(Float64))
    @inbounds ρinv = 1/ρ 
    ∑ρ = 0
    for n = 1:Nspecs
        @inbounds ρn = ρi[i, j, k, n]
        if ρn < 0
            @inbounds ρi[i, j, k, n] = 0
        end
        @inbounds ∑ρ += ρi[i, j, k, n]
    end
    # for n = 1:Nspecs
    #     @inbounds ρi[i, j, k, n] *= ρ/∑ρ
    # end
    @inbounds ρi[i, j, k, Nspecs] += ρ - ∑ρ

    @inbounds u = U[i, j, k, 2]*ρinv # U
    @inbounds v = U[i, j, k, 3]*ρinv # V
    @inbounds w = U[i, j, k, 4]*ρinv # W
    @inbounds ei = max((U[i, j, k, 5] - 0.5*ρ*(u^2 + v^2 + w^2)), CUDA.eps(Float64))

    @inbounds rho = @view ρi[i, j, k, :]
    T::Float64 = max(GetT(ei, rho, thermo), CUDA.eps(Float64))
    p::Float64 = max(Pmixture(T, rho, thermo), CUDA.eps(Float64))

    @inbounds Q[i, j, k, 1] = ρ
    @inbounds Q[i, j, k, 2] = u
    @inbounds Q[i, j, k, 3] = v
    @inbounds Q[i, j, k, 4] = w
    @inbounds Q[i, j, k, 5] = p
    @inbounds Q[i, j, k, 6] = T
    @inbounds Q[i, j, k, 7] = ei
    return
end

# Range: 1 -> N+2*NG
function getY(Yi, ρi, Q)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    @inbounds ρinv::Float64 = 1/max(Q[i, j, k, 1], CUDA.eps(Float64))
    for n = 1:Nspecs
        @inbounds Yi[i, j, k, n] = max(ρi[i, j, k, n]*ρinv, 0.0)
    end
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
