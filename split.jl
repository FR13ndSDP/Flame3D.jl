#For N-S, range: 1->N+2*NG
function fluxSplit(Q, Fp, Fm, Ax, Ay, Az)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    @inbounds ρ = Q[i, j, k, 1]
    @inbounds u = Q[i, j, k, 2]
    @inbounds v = Q[i, j, k, 3]
    @inbounds w = Q[i, j, k, 4]
    @inbounds p = Q[i, j, k, 5]
    @inbounds ei = Q[i, j, k, 7]
    @inbounds A1 = Ax[i, j, k]
    @inbounds A2 = Ay[i, j, k]
    @inbounds A3 = Az[i, j, k]

    γ = p/ei + 1
    c = CUDA.sqrt(γ*p/ρ)
    # γ = 1.4

    ss = CUDA.sqrt(A1*A1 + A2*A2 + A3*A3)
    E1 = A1*u + A2*v + A3*w
    E2 = E1 - c*ss
    E3 = E1 + c*ss

    ss = 1/ss

    A1 *= ss
    A2 *= ss
    A3 *= ss

    E1P = (E1 + CUDA.abs(E1)) * 0.5
    E2P = (E2 + CUDA.abs(E2)) * 0.5
    E3P = (E3 + CUDA.abs(E3)) * 0.5

    E1M = E1 - E1P
    E2M = E2 - E2P
    E3M = E3 - E3P

    uc1 = u - c * A1
    uc2 = u + c * A1
    vc1 = v - c * A2
    vc2 = v + c * A2
    wc1 = w - c * A3
    wc2 = w + c * A3

    vvc1 = (uc1*uc1 + vc1*vc1 + wc1*wc1) * 0.50
    vvc2 = (uc2*uc2 + vc2*vc2 + wc2*wc2) * 0.50
    vv = (γ - 1.0) * (u*u + v*v + w*w)
    W2 = (3-γ)/(2*(γ-1)) * c * c

    tmp1 = ρ/(2 * γ)
    tmp2 = 2 * (γ - 1)
    @inbounds Fp[i, j, k, 1] = tmp1 * (tmp2 * E1P + E2P + E3P)
    @inbounds Fp[i, j, k, 2] = tmp1 * (tmp2 * E1P * u + E2P * uc1 + E3P * uc2)
    @inbounds Fp[i, j, k, 3] = tmp1 * (tmp2 * E1P * v + E2P * vc1 + E3P * vc2)
    @inbounds Fp[i, j, k, 4] = tmp1 * (tmp2 * E1P * w + E2P * wc1 + E3P * wc2)
    @inbounds Fp[i, j, k, 5] = tmp1 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P))

    @inbounds Fm[i, j, k, 1] = tmp1 * (tmp2 * E1M + E2M + E3M)
    @inbounds Fm[i, j, k, 2] = tmp1 * (tmp2 * E1M * u + E2M * uc1 + E3M * uc2)
    @inbounds Fm[i, j, k, 3] = tmp1 * (tmp2 * E1M * v + E2M * vc1 + E3M * vc2)
    @inbounds Fm[i, j, k, 4] = tmp1 * (tmp2 * E1M * w + E2M * wc1 + E3M * wc2)
    @inbounds Fm[i, j, k, 5] = tmp1 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M))
    return 
end

# For species, range 1->N+2*NG
function split(ρi, Q, Fp, Fm, Ax, Ay, Az)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    @inbounds ρ = Q[i, j, k, 1]
    @inbounds u = Q[i, j, k, 2]
    @inbounds v = Q[i, j, k, 3]
    @inbounds w = Q[i, j, k, 4]
    @inbounds p = Q[i, j, k, 5]
    @inbounds ei = Q[i, j, k, 7]
    @inbounds A1 = Ax[i, j, k]
    @inbounds A2 = Ay[i, j, k]
    @inbounds A3 = Az[i, j, k]

    γ = p/ei + 1
    c = CUDA.sqrt(γ*p/ρ)
    # γ = 1.4

    ss = CUDA.sqrt(A1*A1 + A2*A2 + A3*A3)
    E1 = A1*u + A2*v + A3*w
    E2 = E1 - c*ss
    E3 = E1 + c*ss

    ss = 1/ss

    A1 *= ss
    A2 *= ss
    A3 *= ss

    E1P = (E1 + CUDA.abs(E1)) * 0.5
    E2P = (E2 + CUDA.abs(E2)) * 0.5
    E3P = (E3 + CUDA.abs(E3)) * 0.5

    E1M = E1 - E1P
    E2M = E2 - E2P
    E3M = E3 - E3P

    tmp1 = 1/(2 * γ)
    tmp2 = 2 * (γ - 1)

    for n = 1:Nspecs
        @inbounds Fp[i, j, k, n] = tmp1 * (tmp2 * E1P + E2P + E3P) * ρi[i, j, k, n]
        @inbounds Fm[i, j, k, n] = tmp1 * (tmp2 * E1M + E2M + E3M) * ρi[i, j, k, n]
    end
    return
end