# Range: 1+NG -> N+NG
function c2Prim(U, Q)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    # correction
    @inbounds ρ = max(U[i, j, k, 1], CUDA.eps(Float32))
    @inbounds ρinv = 1/ρ 

    @inbounds u = U[i, j, k, 2]*ρinv # U
    @inbounds v = U[i, j, k, 3]*ρinv # V
    @inbounds w = U[i, j, k, 4]*ρinv # W
    @inbounds ei = max((U[i, j, k, 5] - 0.5f0*ρ*(u^2 + v^2 + w^2)), CUDA.eps(Float32))

    p::Float32 = (γ-1) * ei
    T::Float32 = p/(ρ*Rg)

    @inbounds Q[i, j, k, 1] = ρ
    @inbounds Q[i, j, k, 2] = u
    @inbounds Q[i, j, k, 3] = v
    @inbounds Q[i, j, k, 4] = w
    @inbounds Q[i, j, k, 5] = p
    @inbounds Q[i, j, k, 6] = T
    return
end

# Range: 1 -> N+2*NG
function prim2c(U, Q)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
        return
    end

    @inbounds ρ = Q[i, j, k, 1]
    @inbounds u = Q[i, j, k, 2]
    @inbounds v = Q[i, j, k, 3]
    @inbounds w = Q[i, j, k, 4]
    @inbounds ei = Q[i, j, k, 5]/(γ-1)

    @inbounds U[i, j, k, 1] = ρ
    @inbounds U[i, j, k, 2] = u * ρ
    @inbounds U[i, j, k, 3] = v * ρ
    @inbounds U[i, j, k, 4] = w * ρ
    @inbounds U[i, j, k, 5] = ei + 0.5f0 * ρ * (u^2 + v^2 + w^2)
    return
end

# Range: 1+NG -> N+NG
function linComb(U, Un, NV, a::Float32, b::Float32)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    for n = 1:NV
        @inbounds U[i, j, k, n] = U[i, j, k, n] * a + Un[i, j, k, n] * b
    end
    return
end

function pre_x(Q, sc, rth)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    @inbounds p1 = Q[i-2, j, k, 5]
    @inbounds p2 = Q[i-1, j, k, 5]
    @inbounds p3 = Q[i,   j, k, 5]
    @inbounds p4 = Q[i+1, j, k, 5]
    @inbounds p5 = Q[i+2, j, k, 5]

    Δp0 = 0.25f0 * (-p4+2p3-p2)
    Δp1 = 0.25f0 * (-p5+2p4-p3)
    Δp2 = 0.25f0 * (-p3+2p2-p1)
    ri = 0.5f0 * ((Δp0-Δp1)^2+(Δp0-Δp2)^2)/p3^2+1f-16
    @inbounds sc[i, j, k] = 0.5f0*(1.f0-rth/ri+abs(1.f0-rth/ri))
    return
end

function pre_y(Q, sc, rth)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    @inbounds p1 = Q[i, j-2, k, 5]
    @inbounds p2 = Q[i, j-1, k, 5]
    @inbounds p3 = Q[i, j,   k, 5]
    @inbounds p4 = Q[i, j+1, k, 5]
    @inbounds p5 = Q[i, j+2, k, 5]

    Δp0 = 0.25f0 * (-p4+2p3-p2)
    Δp1 = 0.25f0 * (-p5+2p4-p3)
    Δp2 = 0.25f0 * (-p3+2p2-p1)
    ri = 0.5f0 * ((Δp0-Δp1)^2+(Δp0-Δp2)^2)/p3^2+1f-16
    @inbounds sc[i, j, k] = 0.5f0*(1.f0-rth/ri+abs(1.f0-rth/ri))
    return
end

function pre_z(Q, sc, rth)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    @inbounds p1 = Q[i, j, k-2, 5]
    @inbounds p2 = Q[i, j, k-1, 5]
    @inbounds p3 = Q[i, j, k,   5]
    @inbounds p4 = Q[i, j, k+1, 5]
    @inbounds p5 = Q[i, j, k+2, 5]

    Δp0 = 0.25f0 * (-p4+2p3-p2)
    Δp1 = 0.25f0 * (-p5+2p4-p3)
    Δp2 = 0.25f0 * (-p3+2p2-p1)
    ri = 0.5f0 * ((Δp0-Δp1)^2+(Δp0-Δp2)^2)/p3^2+1f-16
    @inbounds sc[i, j, k] = 0.5f0*(1.f0-rth/ri+abs(1.f0-rth/ri))
    return
end

function filter_x(U, Un, sc, s0)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    c1::Float32 = -0.210383f0
    c2::Float32 = 0.039617f0

    @inbounds sc1 = 0.5f0*(sc[i, j, k]+sc[i+1, j, k])
    @inbounds sc2 = 0.5f0*(sc[i, j, k]+sc[i-1, j, k])

    for n = 1:Ncons
        @inbounds U[i, j, k, n] = Un[i, j, k, n] - s0 * (sc1 * (c1 * (Un[i+1, j, k, n] - Un[i, j, k, n]) +
                                                      c2 * (Un[i+2, j, k, n] - Un[i-1, j, k, n])) -
                                               sc2 * (c1 * (Un[i, j, k, n] - Un[i-1, j, k, n]) +
                                                      c2 * (Un[i+1, j, k, n] - Un[i-2, j, k, n])))
    end
    return
end

function filter_y(U, Un, sc, s0)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    c1::Float32 = -0.210383f0
    c2::Float32 = 0.039617f0

    @inbounds sc1 = 0.5f0*(sc[i, j, k]+sc[i, j+1, k])
    @inbounds sc2 = 0.5f0*(sc[i, j, k]+sc[i, j-1, k])

    for n = 1:Ncons
        @inbounds U[i, j, k, n] = Un[i, j, k, n] - s0 * (sc1 * (c1 * (Un[i, j+1, k, n] - Un[i, j, k, n]) +
                                                      c2 * (Un[i, j+2, k, n] - Un[i, j-1, k, n])) -
                                               sc2 * (c1 * (Un[i, j, k, n] - Un[i, j-1, k, n]) +
                                                      c2 * (Un[i, j+1, k, n] - Un[i, j-2, k, n])))
    end
    return
end

function filter_z(U, Un, sc, s0)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    c1::Float32 = -0.210383f0
    c2::Float32 = 0.039617f0

    @inbounds sc1 = 0.5f0*(sc[i, j, k]+sc[i, j, k+1])
    @inbounds sc2 = 0.5f0*(sc[i, j, k]+sc[i, j, k-1])

    for n = 1:Ncons
        @inbounds U[i, j, k, n] = Un[i, j, k, n] - s0 * (sc1 * (c1 * (Un[i, j, k+1, n] - Un[i, j, k, n]) +
                                                      c2 * (Un[i, j, k+2, n] - Un[i, j, k-1, n])) -
                                               sc2 * (c1 * (Un[i, j, k, n] - Un[i, j, k-1, n]) +
                                                      c2 * (Un[i, j, k+1, n] - Un[i, j, k-2, n])))
    end
    return
end

function linearFilter_x(U, Un, s0)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    d0::Float32 = 0.243527493120f0
    d1::Float32 =-0.204788880640f0
    d2::Float32 = 0.120007591680f0
    d3::Float32 =-0.045211119360f0
    d4::Float32 = 0.008228661760f0

    for n = 1:Ncons
        @inbounds U[i, j, k, n] = Un[i, j, k, n] - s0 * (d0 * Un[i, j, k, n] +
                                                         d1 * (Un[i-1, j, k, n] + Un[i+1, j, k, n]) +
                                                         d2 * (Un[i-2, j, k, n] + Un[i+2, j, k, n]) +
                                                         d3 * (Un[i-3, j, k, n] + Un[i+3, j, k, n]) +
                                                         d4 * (Un[i-4, j, k, n] + Un[i+4, j, k, n]))
    end
    return
end

function linearFilter_y(U, Un, s0)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    d0::Float32 = 0.243527493120f0
    d1::Float32 =-0.204788880640f0
    d2::Float32 = 0.120007591680f0
    d3::Float32 =-0.045211119360f0
    d4::Float32 = 0.008228661760f0

    for n = 1:Ncons
        @inbounds U[i, j, k, n] = Un[i, j, k, n] - s0 * (d0 * Un[i, j, k, n] +
                                                         d1 * (Un[i, j-1, k, n] + Un[i, j+1, k, n]) +
                                                         d2 * (Un[i, j-2, k, n] + Un[i, j+2, k, n]) +
                                                         d3 * (Un[i, j-3, k, n] + Un[i, j+3, k, n]) +
                                                         d4 * (Un[i, j-4, k, n] + Un[i, j+4, k, n]))
    end
    return
end

function linearFilter_z(U, Un, s0)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    d0::Float32 = 0.243527493120f0
    d1::Float32 =-0.204788880640f0
    d2::Float32 = 0.120007591680f0
    d3::Float32 =-0.045211119360f0
    d4::Float32 = 0.008228661760f0

    for n = 1:Ncons
        @inbounds U[i, j, k, n] = Un[i, j, k, n] - s0 * (d0 * Un[i, j, k, n] +
                                                         d1 * (Un[i, j, k-1, n] + Un[i, j, k+1, n]) +
                                                         d2 * (Un[i, j, k-2, n] + Un[i, j, k+2, n]) +
                                                         d3 * (Un[i, j, k-3, n] + Un[i, j, k+3, n]) +
                                                         d4 * (Un[i, j, k-4, n] + Un[i, j, k+4, n]))
    end
    return
end