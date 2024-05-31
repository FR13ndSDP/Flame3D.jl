#For N-S, range: 1->N+2*NG
function fluxSplit_SW(Q, Fp, Fm, S, Ax, Ay, Az)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
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
    @inbounds ss = S[i, j, k]

    γ = p/ei + 1
    @fastmath c = sqrt(γ*p/ρ)

    E1 = A1*u + A2*v + A3*w
    E2 = E1 - c*ss
    E3 = E1 + c*ss

    ss = 1/ss

    A1 *= ss
    A2 *= ss
    A3 *= ss

    E1P = (E1 + abs(E1)) * 0.5f0
    E2P = (E2 + abs(E2)) * 0.5f0
    E3P = (E3 + abs(E3)) * 0.5f0

    E1M = E1 - E1P
    E2M = E2 - E2P
    E3M = E3 - E3P

    uc1 = u - c * A1
    uc2 = u + c * A1
    vc1 = v - c * A2
    vc2 = v + c * A2
    wc1 = w - c * A3
    wc2 = w + c * A3

    vvc1 = (uc1^2 + vc1^2 + wc1^2) * 0.5f0
    vvc2 = (uc2^2 + vc2^2 + wc2^2) * 0.5f0
    vv = (γ - 1) * (u^2 + v^2 + w^2)
    W2 = (3-γ)/(2*(γ-1)) * c^2

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

function fluxSplit_LF(Q, Fp, Fm, S, Ax, Ay, Az)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
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
    @inbounds ss = S[i, j, k]

    γ = p/ei + 1
    @fastmath c = sqrt(γ*p/ρ)

    un = A1*u + A2*v + A3*w
    λ0 = abs(un) + c*ss
    E0 = p/(γ-1) + 0.5f0*ρ*(u^2+v^2+w^2)

    @inbounds Fp[i, j, k, 1] = 0.5f0*(ρ*un + λ0*ρ)
    @inbounds Fp[i, j, k, 2] = 0.5f0*(ρ*u*un+A1*p + λ0*ρ*u)
    @inbounds Fp[i, j, k, 3] = 0.5f0*(ρ*v*un+A2*p + λ0*ρ*v)
    @inbounds Fp[i, j, k, 4] = 0.5f0*(ρ*w*un+A3*p + λ0*ρ*w)
    @inbounds Fp[i, j, k, 5] = 0.5f0*((E0+p)*un + λ0*E0)

    @inbounds Fm[i, j, k, 1] = 0.5f0*(ρ*un - λ0*ρ)
    @inbounds Fm[i, j, k, 2] = 0.5f0*(ρ*u*un+A1*p - λ0*ρ*u)
    @inbounds Fm[i, j, k, 3] = 0.5f0*(ρ*v*un+A2*p - λ0*ρ*v)
    @inbounds Fm[i, j, k, 4] = 0.5f0*(ρ*w*un+A3*p - λ0*ρ*w)
    @inbounds Fm[i, j, k, 5] = 0.5f0*((E0+p)*un - λ0*E0)
    return 
end

function fluxSplit_VL(Q, Fp, Fm, S, Ax, Ay, Az)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
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
    @inbounds ss = S[i, j, k]

    γ = p/ei + 1
    @fastmath c = sqrt(γ*p/ρ)

    un = A1*u + A2*v + A3*w
    E0 = p/(γ-1) + 0.5f0*ρ*(u^2+v^2+w^2)
    
    M = un/(c*ss)

    if M <= -1
        @inbounds Fp[i, j, k, 1] = 0.f0
        @inbounds Fp[i, j, k, 2] = 0.f0
        @inbounds Fp[i, j, k, 3] = 0.f0
        @inbounds Fp[i, j, k, 4] = 0.f0
        @inbounds Fp[i, j, k, 5] = 0.f0
    
        @inbounds Fm[i, j, k, 1] = ρ * un
        @inbounds Fm[i, j, k, 2] = ρ * u * un + A1 * p
        @inbounds Fm[i, j, k, 3] = ρ * v * un + A2 * p
        @inbounds Fm[i, j, k, 4] = ρ * w * un + A3 * p
        @inbounds Fm[i, j, k, 5] = (E0 + p) * un
    elseif M >= 1
        @inbounds Fm[i, j, k, 1] = 0.f0
        @inbounds Fm[i, j, k, 2] = 0.f0
        @inbounds Fm[i, j, k, 3] = 0.f0
        @inbounds Fm[i, j, k, 4] = 0.f0
        @inbounds Fm[i, j, k, 5] = 0.f0
    
        @inbounds Fp[i, j, k, 1] = ρ * un
        @inbounds Fp[i, j, k, 2] = ρ * u * un + A1 * p
        @inbounds Fp[i, j, k, 3] = ρ * v * un + A2 * p
        @inbounds Fp[i, j, k, 4] = ρ * w * un + A3 * p
        @inbounds Fp[i, j, k, 5] = (E0 + p) * un
    else
        un /= ss
        f1p = ρ * c * 0.25f0 *(M + 1)^2 * ss
        f1m = -ρ * c * 0.25f0 *(M - 1)^2 * ss
        tmp1 = (-un+2*c)/(γ*ss)
        tmp2 = (-un-2*c)/(γ*ss)
        tmp3 = 1/(2*(γ^2-1))
        tmp4 = 0.5f0*(u^2+v^2+w^2-un^2)

        @inbounds Fp[i, j, k, 1] = f1p
        @inbounds Fp[i, j, k, 2] = f1p*(A1*tmp1 + u)
        @inbounds Fp[i, j, k, 3] = f1p*(A2*tmp1 + v)
        @inbounds Fp[i, j, k, 4] = f1p*(A3*tmp1 + w)
        @inbounds Fp[i, j, k, 5] = f1p*(((γ-1)*un+2*c)^2*tmp3+tmp4)
    
        @inbounds Fm[i, j, k, 1] = f1m
        @inbounds Fm[i, j, k, 2] = f1m*(A1*tmp2 + u)
        @inbounds Fm[i, j, k, 3] = f1m*(A2*tmp2 + v)
        @inbounds Fm[i, j, k, 4] = f1m*(A3*tmp2 + w)
        @inbounds Fm[i, j, k, 5] = f1m*(((γ-1)*un-2*c)^2*tmp3+tmp4)
    end
    return
end

function fluxSplit_AUSM(Q, Fp, Fm, S, Ax, Ay, Az)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
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
    @inbounds ss = S[i, j, k]

    γ = p/ei + 1
    @fastmath c = sqrt(γ*p/ρ)

    un = A1*u + A2*v + A3*w
    h = γ*p/(ρ*(γ-1)) + 0.5f0*(u^2+v^2+w^2)
    
    M = un/(c*ss)

    if M <= -1
        @inbounds Fp[i, j, k, 1] = 0.f0
        @inbounds Fp[i, j, k, 2] = 0.f0
        @inbounds Fp[i, j, k, 3] = 0.f0
        @inbounds Fp[i, j, k, 4] = 0.f0
        @inbounds Fp[i, j, k, 5] = 0.f0
    
        @inbounds Fm[i, j, k, 1] = ρ * un
        @inbounds Fm[i, j, k, 2] = ρ * u * un + A1 * p
        @inbounds Fm[i, j, k, 3] = ρ * v * un + A2 * p
        @inbounds Fm[i, j, k, 4] = ρ * w * un + A3 * p
        @inbounds Fm[i, j, k, 5] = ρ * h * un
    elseif M >= 1
        @inbounds Fm[i, j, k, 1] = 0.f0
        @inbounds Fm[i, j, k, 2] = 0.f0
        @inbounds Fm[i, j, k, 3] = 0.f0
        @inbounds Fm[i, j, k, 4] = 0.f0
        @inbounds Fm[i, j, k, 5] = 0.f0
    
        @inbounds Fp[i, j, k, 1] = ρ * un
        @inbounds Fp[i, j, k, 2] = ρ * u * un + A1 * p
        @inbounds Fp[i, j, k, 3] = ρ * v * un + A2 * p
        @inbounds Fp[i, j, k, 4] = ρ * w * un + A3 * p
        @inbounds Fp[i, j, k, 5] = ρ * h * un
    else
        fp = 0.25f0*(M + 1)^2*ρ*c*ss
        fm = -0.25f0*(M - 1)^2*ρ*c*ss

        Pp = p * (0.25f0*(M+1)^2*(2-M))
        Pm = p * (0.25f0*(M-1)^2*(2+M))

        @inbounds Fp[i, j, k, 1] = fp
        @inbounds Fp[i, j, k, 2] = fp * u + A1 * Pp
        @inbounds Fp[i, j, k, 3] = fp * v + A2 * Pp
        @inbounds Fp[i, j, k, 4] = fp * w + A3 * Pp
        @inbounds Fp[i, j, k, 5] = fp * h
    
        @inbounds Fm[i, j, k, 1] = fm
        @inbounds Fm[i, j, k, 2] = fm * u + A1 * Pm
        @inbounds Fm[i, j, k, 3] = fm * v + A2 * Pm
        @inbounds Fm[i, j, k, 4] = fm * w + A3 * Pm
        @inbounds Fm[i, j, k, 5] = fm * h
    end
    return
end

"""
    split(ρi, Q, Fp, Fm, Ax, Ay, Az)

Do flux-vector splitting on grid points for species (include ghosts).

...
# Notes
- Species treated as scalar so only advect with velocity, the split is 1/2(U±|U|)
...
"""
function split(ρi, Q, Fp, Fm, Ax, Ay, Az)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
        return
    end

    @inbounds u = Q[i, j, k, 2]
    @inbounds v = Q[i, j, k, 3]
    @inbounds w = Q[i, j, k, 4]
    @inbounds A1 = Ax[i, j, k]
    @inbounds A2 = Ay[i, j, k]
    @inbounds A3 = Az[i, j, k]

    un = A1*u + A2*v + A3*w
    Ep = 0.5f0 * (un + abs(un))
    Em = 0.5f0 * (un - abs(un))

    for n = 1:Nspecs
        @inbounds Fp[i, j, k, n] = Ep * ρi[i, j, k, n]
        @inbounds Fm[i, j, k, n] = Em * ρi[i, j, k, n]
    end

    return
end