function shockSensor(ϕ, Q)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i < 2 || i > Nxp+2*NG-1 || j < 2 || j > Ny+2*NG-1 || k < 2 || k > Nz+2*NG-1
        return
    end

    @inbounds Px1 = Q[i-1, j, k, 5]
    @inbounds Px2 = Q[i,   j, k, 5]
    @inbounds Px3 = Q[i+1, j, k, 5]
    @inbounds Py1 = Q[i, j-1, k, 5]
    @inbounds Py2 = Q[i, j,   k, 5]
    @inbounds Py3 = Q[i, j+1, k, 5]
    @inbounds Pz1 = Q[i, j, k-1, 5]
    @inbounds Pz2 = Q[i, j, k  , 5]
    @inbounds Pz3 = Q[i, j, k+1, 5]
    ϕx = abs(-Px1 + 2*Px2 - Px3)/(Px1 + 2*Px2 + Px3)
    ϕy = abs(-Py1 + 2*Py2 - Py3)/(Py1 + 2*Py2 + Py3)
    ϕz = abs(-Pz1 + 2*Pz2 - Pz3)/(Pz1 + 2*Pz2 + Pz3)
    @inbounds ϕ[i, j, k] = ϕx + ϕy + ϕz
    return
end

@inline function minmod(a, b)
    ifelse(a*b > 0, (abs(a) > abs(b)) ? b : a, zero(a))
end

#Range: 1 -> N+1
function NND_x(F, Fp, Fm, NV)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG+1 || j > Ny+NG || k > Nz+NG || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end
    for n = 1:NV
        @inbounds fp = Fp[i-1, j, k, n] + 0.5*minmod(Fp[i, j, k, n] - Fp[i-1, j, k, n], 
                                                     Fp[i-1, j, k, n] - Fp[i-2, j, k, n])
        @inbounds fm = Fm[i, j, k, n] - 0.5*minmod(Fm[i+1, j, k, n] - Fm[i, j, k, n], 
                                                   Fm[i, j, k, n] - Fm[i-1, j, k, n])
        @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
    end
    return
end

function NND_y(F, Fp, Fm, NV)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Ny+NG+1 || k > Nz+NG || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end
    for n = 1:NV
        @inbounds fp = Fp[i, j-1, k, n] + 0.5*minmod(Fp[i, j, k, n] - Fp[i, j-1, k, n], 
                                                     Fp[i, j-1, k, n] - Fp[i, j-2, k, n])
        @inbounds fm = Fm[i, j, k, n] - 0.5*minmod(Fm[i, j+1, k, n] - Fm[i, j, k, n], 
                                                   Fm[i, j, k, n] - Fm[i, j-1, k, n])
        @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
    end
    return
end

function NND_z(F, Fp, Fm, NV)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Ny+NG || k > Nz+NG+1 || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end
    for n = 1:NV
        @inbounds fp = Fp[i, j, k-1, n] + 0.5*minmod(Fp[i, j, k, n] - Fp[i, j, k-1, n], 
                                                     Fp[i, j, k-1, n] - Fp[i, j, k-2, n])
        @inbounds fm = Fm[i, j, k, n] - 0.5*minmod(Fm[i, j, k+1, n] - Fm[i, j, k, n], 
                                                   Fm[i, j, k, n] - Fm[i, j, k-1, n])
        @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
    end
    return
end

#Range: 1 -> N+1
function WENO_x(F, ϕ, Fp, Fm, NV, consts)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG+1 || j > Ny+NG || k > Nz+NG || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end

    eps::Float64 = consts.WENO5[1]
    tmp1::Float64 = consts.WENO5[2]
    tmp2::Float64 = consts.WENO5[3]

    c1::Float64 = consts.UP7[1]
    c2::Float64 = consts.UP7[2]
    c3::Float64 = consts.UP7[3]
    c4::Float64 = consts.UP7[4]
    c5::Float64 = consts.UP7[5]
    c6::Float64 = consts.UP7[6]
    c7::Float64 = consts.UP7[7]

    # Jameson sensor
    @inbounds ϕx = max(ϕ[i-1, j, k], ϕ[i, j, k])

    if ϕx < consts.Hybrid[1]
        for n = 1:NV
            @inbounds V1 = Fp[i-4, j, k, n]
            @inbounds V2 = Fp[i-3, j, k, n]
            @inbounds V3 = Fp[i-2, j, k, n]
            @inbounds V4 = Fp[i-1, j, k, n]
            @inbounds V5 = Fp[i,   j, k, n]
            @inbounds V6 = Fp[i+1, j, k, n]
            @inbounds V7 = Fp[i+2, j, k, n]
            
            fp = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6 + c7*V7

            @inbounds V1 = Fm[i-3, j, k, n]
            @inbounds V2 = Fm[i-2, j, k, n]
            @inbounds V3 = Fm[i-1, j, k, n]
            @inbounds V4 = Fm[i,   j, k, n]
            @inbounds V5 = Fm[i+1, j, k, n]
            @inbounds V6 = Fm[i+2, j, k, n]
            @inbounds V7 = Fm[i+3, j, k, n]

            fm = c7*V1 + c6*V2 + c5*V3 + c4*V4 + c3*V5 + c2*V6 + c1*V7

            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end
    elseif ϕx < consts.Hybrid[2]
        for n = 1:NV
            @inbounds V1 = Fp[i-3, j, k, n]
            @inbounds V2 = Fp[i-2, j, k, n]
            @inbounds V3 = Fp[i-1, j, k, n]
            @inbounds V4 = Fp[i,   j, k, n]
            @inbounds V5 = Fp[i+1, j, k, n]
            # FP
            s11 = tmp1*(V1-2*V2+V3)^2 + 0.25*(V1-4*V2+3*V3)^2
            s22 = tmp1*(V2-2*V3+V4)^2 + 0.25*(V2-V4)^2
            s33 = tmp1*(V3-2*V4+V5)^2 + 0.25*(3*V3-4*V4+V5)^2

            # s11 = 0.1/(eps+s11)^2
            # s22 = 0.6/(eps+s22)^2
            # s33 = 0.3/(eps+s33)^2
            τ = abs(s11-s33)
            s11 = (1 + (τ/(eps+s11))^2) * 1
            s22 = (1 + (τ/(eps+s22))^2) * 6
            s33 = (1 + (τ/(eps+s33))^2) * 3

            invsum = 1/(s11+s22+s33)

            v1 = 2*V1-7*V2+11*V3
            v2 = -V2+5*V3+2*V4
            v3 = 2*V3+5*V4-V5
            fp = tmp2*invsum*(s11*v1+s22*v2+s33*v3)

            @inbounds V1 = Fm[i-2, j, k, n]
            @inbounds V2 = Fm[i-1, j, k, n]
            @inbounds V3 = Fm[i,   j, k, n]
            @inbounds V4 = Fm[i+1, j, k, n]
            @inbounds V5 = Fm[i+2, j, k, n]
            # FM
            s11 = tmp1*(V5-2*V4+V3)^2 + 0.25*(V5-4*V4+3*V3)^2
            s22 = tmp1*(V2-2*V3+V4)^2 + 0.25*(V4-V2)^2
            s33 = tmp1*(V3-2*V2+V1)^2 + 0.25*(3*V3-4*V2+V1)^2

            # s11 = 0.1/(eps+s11)^2
            # s22 = 0.6/(eps+s22)^2
            # s33 = 0.3/(eps+s33)^2
            τ = abs(s11-s33)
            s11 = (1 + (τ/(eps+s11))^2) * 1
            s22 = (1 + (τ/(eps+s22))^2) * 6
            s33 = (1 + (τ/(eps+s33))^2) * 3

            invsum = 1/(s11+s22+s33)

            v1 = 11*V3-7*V4+2*V5
            v2 = -V4+5*V3+2*V2
            v3 = 2*V3+5*V2-V1
            fm = tmp2*invsum*(s11*v1+s22*v2+s33*v3)
            
            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end
    else
        for n = 1:NV
            @inbounds fp = Fp[i-1, j, k, n] + 0.5*minmod(Fp[i, j, k, n] - Fp[i-1, j, k, n], 
                                                         Fp[i-1, j, k, n] - Fp[i-2, j, k, n])
            @inbounds fm = Fm[i, j, k, n] - 0.5*minmod(Fm[i+1, j, k, n] - Fm[i, j, k, n], 
                                                       Fm[i, j, k, n] - Fm[i-1, j, k, n])
            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end
    end
    return
end

#Range: 1 -> N+1
function WENO_y(F, ϕ, Fp, Fm, NV, consts)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Ny+NG+1 || k > Nz+NG || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end

    eps::Float64 = consts.WENO5[1]
    tmp1::Float64 = consts.WENO5[2]
    tmp2::Float64 = consts.WENO5[3]

    c1::Float64 = consts.UP7[1]
    c2::Float64 = consts.UP7[2]
    c3::Float64 = consts.UP7[3]
    c4::Float64 = consts.UP7[4]
    c5::Float64 = consts.UP7[5]
    c6::Float64 = consts.UP7[6]
    c7::Float64 = consts.UP7[7]

    # Jameson sensor
    @inbounds ϕy = max(ϕ[i, j-1, k], ϕ[i, j, k])

    if ϕy < consts.Hybrid[1]
        for n = 1:NV
            @inbounds V1 = Fp[i, j-4, k, n]
            @inbounds V2 = Fp[i, j-3, k, n]
            @inbounds V3 = Fp[i, j-2, k, n]
            @inbounds V4 = Fp[i, j-1, k, n]
            @inbounds V5 = Fp[i, j,   k, n]
            @inbounds V6 = Fp[i, j+1, k, n]
            @inbounds V7 = Fp[i, j+2, k, n]
            
            fp = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6 + c7*V7

            @inbounds V1 = Fm[i, j-3, k, n]
            @inbounds V2 = Fm[i, j-2, k, n]
            @inbounds V3 = Fm[i, j-1, k, n]
            @inbounds V4 = Fm[i, j,   k, n]
            @inbounds V5 = Fm[i, j+1, k, n]
            @inbounds V6 = Fm[i, j+2, k, n]
            @inbounds V7 = Fm[i, j+3, k, n]

            fm = c7*V1 + c6*V2 + c5*V3 + c4*V4 + c3*V5 + c2*V6 + c1*V7

            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end
    elseif ϕy < consts.Hybrid[2]
        for n = 1:NV
            @inbounds V1 = Fp[i, j-3, k, n]
            @inbounds V2 = Fp[i, j-2, k, n]
            @inbounds V3 = Fp[i, j-1, k, n]
            @inbounds V4 = Fp[i, j,   k, n]
            @inbounds V5 = Fp[i, j+1, k, n]
            # FP
            s11 = tmp1*(V1-2*V2+V3)^2 + 0.25*(V1-4*V2+3*V3)^2
            s22 = tmp1*(V2-2*V3+V4)^2 + 0.25*(V2-V4)^2
            s33 = tmp1*(V3-2*V4+V5)^2 + 0.25*(3*V3-4*V4+V5)^2

            # s11 = 0.1/(eps+s11)^2
            # s22 = 0.6/(eps+s22)^2
            # s33 = 0.3/(eps+s33)^2
            τ = abs(s11-s33)
            s11 = (1 + (τ/(eps+s11))^2) * 1
            s22 = (1 + (τ/(eps+s22))^2) * 6
            s33 = (1 + (τ/(eps+s33))^2) * 3

            invsum = 1/(s11+s22+s33)

            v1 = 2*V1-7*V2+11*V3
            v2 = -V2+5*V3+2*V4
            v3 = 2*V3+5*V4-V5
            fp = tmp2*invsum*(s11*v1+s22*v2+s33*v3)

            @inbounds V1 = Fm[i, j-2, k, n]
            @inbounds V2 = Fm[i, j-1, k, n]
            @inbounds V3 = Fm[i, j,   k, n]
            @inbounds V4 = Fm[i, j+1, k, n]
            @inbounds V5 = Fm[i, j+2, k, n]
            # FM
            s11 = tmp1*(V5-2*V4+V3)^2 + 0.25*(V5-4*V4+3*V3)^2
            s22 = tmp1*(V2-2*V3+V4)^2 + 0.25*(V4-V2)^2
            s33 = tmp1*(V3-2*V2+V1)^2 + 0.25*(3*V3-4*V2+V1)^2

            # s11 = 0.1/(eps+s11)^2
            # s22 = 0.6/(eps+s22)^2
            # s33 = 0.3/(eps+s33)^2
            τ = abs(s11-s33)
            s11 = (1 + (τ/(eps+s11))^2) * 1
            s22 = (1 + (τ/(eps+s22))^2) * 6
            s33 = (1 + (τ/(eps+s33))^2) * 3

            invsum = 1/(s11+s22+s33)

            v1 = 11*V3-7*V4+2*V5
            v2 = -V4+5*V3+2*V2
            v3 = 2*V3+5*V2-V1
            fm = tmp2*invsum*(s11*v1+s22*v2+s33*v3)
            
            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end
    else
        for n = 1:NV
            @inbounds fp = Fp[i, j-1, k, n] + 0.5*minmod(Fp[i, j, k, n] - Fp[i, j-1, k, n], 
                                                         Fp[i, j-1, k, n] - Fp[i, j-2, k, n])
            @inbounds fm = Fm[i, j, k, n] - 0.5*minmod(Fm[i, j+1, k, n] - Fm[i, j, k, n], 
                                                       Fm[i, j, k, n] - Fm[i, j-1, k, n])
            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end
    end
    return
end

#Range: 1 -> N+1
function WENO_z(F, ϕ, Fp, Fm, NV, consts)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Ny+NG || k > Nz+NG+1 || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end

    eps::Float64 = consts.WENO5[1]
    tmp1::Float64 = consts.WENO5[2]
    tmp2::Float64 = consts.WENO5[3]

    c1::Float64 = consts.UP7[1]
    c2::Float64 = consts.UP7[2]
    c3::Float64 = consts.UP7[3]
    c4::Float64 = consts.UP7[4]
    c5::Float64 = consts.UP7[5]
    c6::Float64 = consts.UP7[6]
    c7::Float64 = consts.UP7[7]

    # Jameson sensor
    @inbounds ϕz = max(ϕ[i, j, k-1], ϕ[i, j, k])

    if ϕz < consts.Hybrid[1]
        for n = 1:NV
            @inbounds V1 = Fp[i, j, k-4, n]
            @inbounds V2 = Fp[i, j, k-3, n]
            @inbounds V3 = Fp[i, j, k-2, n]
            @inbounds V4 = Fp[i, j, k-1, n]
            @inbounds V5 = Fp[i, j, k,   n]
            @inbounds V6 = Fp[i, j, k+1, n]
            @inbounds V7 = Fp[i, j, k+2, n]
            
            fp = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6 + c7*V7

            @inbounds V1 = Fm[i, j, k-3, n]
            @inbounds V2 = Fm[i, j, k-2, n]
            @inbounds V3 = Fm[i, j, k-1, n]
            @inbounds V4 = Fm[i, j, k,   n]
            @inbounds V5 = Fm[i, j, k+1, n]
            @inbounds V6 = Fm[i, j, k+2, n]
            @inbounds V7 = Fm[i, j, k+3, n]

            fm = c7*V1 + c6*V2 + c5*V3 + c4*V4 + c3*V5 + c2*V6 + c1*V7

            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end
    elseif ϕz < consts.Hybrid[2]
        for n = 1:NV
            @inbounds V1 = Fp[i, j, k-3, n]
            @inbounds V2 = Fp[i, j, k-2, n]
            @inbounds V3 = Fp[i, j, k-1, n]
            @inbounds V4 = Fp[i, j, k,   n]
            @inbounds V5 = Fp[i, j, k+1, n]
            # FP
            s11 = tmp1*(V1-2*V2+V3)^2 + 0.25*(V1-4*V2+3*V3)^2
            s22 = tmp1*(V2-2*V3+V4)^2 + 0.25*(V2-V4)^2
            s33 = tmp1*(V3-2*V4+V5)^2 + 0.25*(3*V3-4*V4+V5)^2

            # s11 = 0.1/(eps+s11)^2
            # s22 = 0.6/(eps+s22)^2
            # s33 = 0.3/(eps+s33)^2
            τ = abs(s11-s33)
            s11 = (1 + (τ/(eps+s11))^2) * 1
            s22 = (1 + (τ/(eps+s22))^2) * 6
            s33 = (1 + (τ/(eps+s33))^2) * 3

            invsum = 1/(s11+s22+s33)

            v1 = 2*V1-7*V2+11*V3
            v2 = -V2+5*V3+2*V4
            v3 = 2*V3+5*V4-V5
            fp = tmp2*invsum*(s11*v1+s22*v2+s33*v3)

            @inbounds V1 = Fm[i, j, k-2, n]
            @inbounds V2 = Fm[i, j, k-1, n]
            @inbounds V3 = Fm[i, j, k,   n]
            @inbounds V4 = Fm[i, j, k+1, n]
            @inbounds V5 = Fm[i, j, k+2, n]
            # FM
            s11 = tmp1*(V5-2*V4+V3)^2 + 0.25*(V5-4*V4+3*V3)^2
            s22 = tmp1*(V2-2*V3+V4)^2 + 0.25*(V4-V2)^2
            s33 = tmp1*(V3-2*V2+V1)^2 + 0.25*(3*V3-4*V2+V1)^2

            # s11 = 0.1/(eps+s11)^2
            # s22 = 0.6/(eps+s22)^2
            # s33 = 0.3/(eps+s33)^2
            τ = abs(s11-s33)
            s11 = (1 + (τ/(eps+s11))^2) * 1
            s22 = (1 + (τ/(eps+s22))^2) * 6
            s33 = (1 + (τ/(eps+s33))^2) * 3

            invsum = 1/(s11+s22+s33)

            v1 = 11*V3-7*V4+2*V5
            v2 = -V4+5*V3+2*V2
            v3 = 2*V3+5*V2-V1
            fm = tmp2*invsum*(s11*v1+s22*v2+s33*v3)
            
            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end
    else
        for n = 1:NV
            @inbounds fp = Fp[i, j, k-1, n] + 0.5*minmod(Fp[i, j, k, n] - Fp[i, j, k-1, n], 
                                                         Fp[i, j, k-1, n] - Fp[i, j, k-2, n])
            @inbounds fm = Fm[i, j, k, n] - 0.5*minmod(Fm[i, j, k+1, n] - Fm[i, j, k, n], 
                                                       Fm[i, j, k, n] - Fm[i, j, k-1, n])
            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end  
    end
    return
end

function TENO_x(F, Fp, Fm, NV, consts)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG+1 || j > Ny+NG || k > Nz+NG || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end

    eps::Float64 = consts.TENO5[1]
    CT::Float64 = consts.TENO5[2]
    oneSix::Float64 = consts.TENO5[3]

    for n = 1:NV
        @inbounds V1 = Fp[i-3, j, k, n]
        @inbounds V2 = Fp[i-2, j, k, n]
        @inbounds V3 = Fp[i-1, j, k, n]
        @inbounds V4 = Fp[i,   j, k, n]
        @inbounds V5 = Fp[i+1, j, k, n]
        # FP
        s11 = 13*(V1-2*V2+V3)^2 + 3*(V1-4*V2+3*V3)^2
        s22 = 13*(V2-2*V3+V4)^2 + 3*(V2-V4)^2
        s33 = 13*(V3-2*V4+V5)^2 + 3*(3*V3-4*V4+V5)^2

        τ = abs(s11-s33)
        @fastmath a1 = (1 + τ/(s11+eps))^6
        @fastmath a2 = (1 + τ/(s22+eps))^6
        @fastmath a3 = (1 + τ/(s33+eps))^6
        
        invsum = 1/(a1+a2+a3)
        
        a1 = 1*((a1*invsum)<CT ? 0.0 : 1.0)
        a2 = 6*((a2*invsum)<CT ? 0.0 : 1.0)
        a3 = 3*((a3*invsum)<CT ? 0.0 : 1.0)
        
        v1 = 2*V1-7*V2+5*V3
        v2 = -V2-V3+2*V4
        v3 = -4*V3+5*V4-V5
        
        invsum = 1/(a1+a2+a3)
        
        fp = V3+oneSix*invsum*(a1*v1+a2*v2+a3*v3)

        @inbounds V1 = Fm[i-2, j, k, n]
        @inbounds V2 = Fm[i-1, j, k, n]
        @inbounds V3 = Fm[i,   j, k, n]
        @inbounds V4 = Fm[i+1, j, k, n]
        @inbounds V5 = Fm[i+2, j, k, n]
        # FM
        s11 = 13*(V5-2*V4+V3)^2 + 3*(V5-4*V4+3*V3)^2
        s22 = 13*(V2-2*V3+V4)^2 + 3*(V4-V2)^2
        s33 = 13*(V3-2*V2+V1)^2 + 3*(3*V3-4*V2+V1)^2

        τ = abs(s11-s33)
        @fastmath a1 = (1 + τ/(s11+eps))^6
        @fastmath a2 = (1 + τ/(s22+eps))^6
        @fastmath a3 = (1 + τ/(s33+eps))^6
        
        invsum = 1/(a1+a2+a3)
        
        a1 = 1*((a1*invsum)<1e-5 ? 0.0 : 1.0)
        a2 = 6*((a2*invsum)<1e-5 ? 0.0 : 1.0)
        a3 = 3*((a3*invsum)<1e-5 ? 0.0 : 1.0)
        
        v1 = 5*V3-7*V4+2*V5
        v2 = -V4-V3+2*V2
        v3 = -4*V3+5*V2-V1
        
        invsum = 1/(a1+a2+a3)
        
        fm = V3+oneSix*invsum*(a1*v1+a2*v2+a3*v3)
        
        @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
    end
    return
end

function TENO_y(F, Fp, Fm, NV, consts)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Ny+NG+1 || k > Nz+NG || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end

    eps::Float64 = consts.TENO5[1]
    CT::Float64 = consts.TENO5[2]
    oneSix::Float64 = consts.TENO5[3]

    for n = 1:NV
        @inbounds V1 = Fp[i, j-3, k, n]
        @inbounds V2 = Fp[i, j-2, k, n]
        @inbounds V3 = Fp[i, j-1, k, n]
        @inbounds V4 = Fp[i, j,   k, n]
        @inbounds V5 = Fp[i, j+1, k, n]
        # FP
        s11 = 13*(V1-2*V2+V3)^2 + 3*(V1-4*V2+3*V3)^2
        s22 = 13*(V2-2*V3+V4)^2 + 3*(V2-V4)^2
        s33 = 13*(V3-2*V4+V5)^2 + 3*(3*V3-4*V4+V5)^2

        τ = abs(s11-s33)
        @fastmath a1 = (1 + τ/(s11+eps))^6
        @fastmath a2 = (1 + τ/(s22+eps))^6
        @fastmath a3 = (1 + τ/(s33+eps))^6
        
        invsum = 1/(a1+a2+a3)
        
        a1 = 1*((a1*invsum)<CT ? 0.0 : 1.0)
        a2 = 6*((a2*invsum)<CT ? 0.0 : 1.0)
        a3 = 3*((a3*invsum)<CT ? 0.0 : 1.0)
        
        v1 = 2*V1-7*V2+5*V3
        v2 = -V2-V3+2*V4
        v3 = -4*V3+5*V4-V5
        
        invsum = 1/(a1+a2+a3)
        
        fp = V3+oneSix*invsum*(a1*v1+a2*v2+a3*v3)

        @inbounds V1 = Fm[i, j-2, k, n]
        @inbounds V2 = Fm[i, j-1, k, n]
        @inbounds V3 = Fm[i, j,   k, n]
        @inbounds V4 = Fm[i, j+1, k, n]
        @inbounds V5 = Fm[i, j+2, k, n]
        # FM
        s11 = 13*(V5-2*V4+V3)^2 + 3*(V5-4*V4+3*V3)^2
        s22 = 13*(V2-2*V3+V4)^2 + 3*(V4-V2)^2
        s33 = 13*(V3-2*V2+V1)^2 + 3*(3*V3-4*V2+V1)^2

        τ = abs(s11-s33)
        @fastmath a1 = (1 + τ/(s11+eps))^6
        @fastmath a2 = (1 + τ/(s22+eps))^6
        @fastmath a3 = (1 + τ/(s33+eps))^6
        
        invsum = 1/(a1+a2+a3)
        
        a1 = 1*((a1*invsum)<1e-5 ? 0.0 : 1.0)
        a2 = 6*((a2*invsum)<1e-5 ? 0.0 : 1.0)
        a3 = 3*((a3*invsum)<1e-5 ? 0.0 : 1.0)
        
        v1 = 5*V3-7*V4+2*V5
        v2 = -V4-V3+2*V2
        v3 = -4*V3+5*V2-V1
        
        invsum = 1/(a1+a2+a3)
        
        fm = V3+oneSix*invsum*(a1*v1+a2*v2+a3*v3)
        
        @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
    end
    return
end

function TENO_z(F, Fp, Fm, NV, consts)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Ny+NG || k > Nz+NG+1 || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end

    eps::Float64 = consts.TENO5[1]
    CT::Float64 = consts.TENO5[2]
    oneSix::Float64 = consts.TENO5[3]

    for n = 1:NV
        @inbounds V1 = Fp[i, j, k-3, n]
        @inbounds V2 = Fp[i, j, k-2, n]
        @inbounds V3 = Fp[i, j, k-1, n]
        @inbounds V4 = Fp[i, j, k,   n]
        @inbounds V5 = Fp[i, j, k+1, n]
        # FP
        s11 = 13*(V1-2*V2+V3)^2 + 3*(V1-4*V2+3*V3)^2
        s22 = 13*(V2-2*V3+V4)^2 + 3*(V2-V4)^2
        s33 = 13*(V3-2*V4+V5)^2 + 3*(3*V3-4*V4+V5)^2

        τ = abs(s11-s33)
        @fastmath a1 = (1 + τ/(s11+eps))^6
        @fastmath a2 = (1 + τ/(s22+eps))^6
        @fastmath a3 = (1 + τ/(s33+eps))^6
        
        invsum = 1/(a1+a2+a3)
        
        a1 = 1*((a1*invsum)<CT ? 0.0 : 1.0)
        a2 = 6*((a2*invsum)<CT ? 0.0 : 1.0)
        a3 = 3*((a3*invsum)<CT ? 0.0 : 1.0)
        
        v1 = 2*V1-7*V2+5*V3
        v2 = -V2-V3+2*V4
        v3 = -4*V3+5*V4-V5
        
        invsum = 1/(a1+a2+a3)
        
        fp = V3+oneSix*invsum*(a1*v1+a2*v2+a3*v3)

        @inbounds V1 = Fm[i, j, k-2, n]
        @inbounds V2 = Fm[i, j, k-1, n]
        @inbounds V3 = Fm[i, j, k,   n]
        @inbounds V4 = Fm[i, j, k+1, n]
        @inbounds V5 = Fm[i, j, k+2, n]
        # FM
        s11 = 13*(V5-2*V4+V3)^2 + 3*(V5-4*V4+3*V3)^2
        s22 = 13*(V2-2*V3+V4)^2 + 3*(V4-V2)^2
        s33 = 13*(V3-2*V2+V1)^2 + 3*(3*V3-4*V2+V1)^2

        τ = abs(s11-s33)
        @fastmath a1 = (1 + τ/(s11+eps))^6
        @fastmath a2 = (1 + τ/(s22+eps))^6
        @fastmath a3 = (1 + τ/(s33+eps))^6
        
        invsum = 1/(a1+a2+a3)
        
        a1 = 1*((a1*invsum)<1e-5 ? 0.0 : 1.0)
        a2 = 6*((a2*invsum)<1e-5 ? 0.0 : 1.0)
        a3 = 3*((a3*invsum)<1e-5 ? 0.0 : 1.0)
        
        v1 = 5*V3-7*V4+2*V5
        v2 = -V4-V3+2*V2
        v3 = -4*V3+5*V2-V1
        
        invsum = 1/(a1+a2+a3)
        
        fm = V3+oneSix*invsum*(a1*v1+a2*v2+a3*v3)
        
        @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
    end
    return
end