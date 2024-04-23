# detect shock and discontinuity, by Antony Jameson for JST scheme
function shockSensor(ϕ, Q)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i < 2 || i > Nxp+2*NG-1 || j < 2 || j > Nyp+2*NG-1 || k < 2 || k > Nzp+2*NG-1
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

# inline function for NND
@inline function minmod(a, b)
    ifelse(a*b > 0, (abs(a) > abs(b)) ? b : a, zero(a))
end

#= 
    2nd order NND
    For debug
=#
function NND_x(F, Fp, Fm, NV)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+NG+1 || j > Nyp+NG || k > Nzp+NG || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end
    for n = 1:NV
        @inbounds fp = Fp[i-1, j, k, n] + 0.5f0*minmod(Fp[i, j, k, n] - Fp[i-1, j, k, n], 
                                                     Fp[i-1, j, k, n] - Fp[i-2, j, k, n])
        @inbounds fm = Fm[i, j, k, n] - 0.5f0*minmod(Fm[i+1, j, k, n] - Fm[i, j, k, n], 
                                                   Fm[i, j, k, n] - Fm[i-1, j, k, n])
        @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
    end
    return
end

function NND_y(F, Fp, Fm, NV)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+NG || j > Nyp+NG+1 || k > Nzp+NG || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end
    for n = 1:NV
        @inbounds fp = Fp[i, j-1, k, n] + 0.5f0*minmod(Fp[i, j, k, n] - Fp[i, j-1, k, n], 
                                                     Fp[i, j-1, k, n] - Fp[i, j-2, k, n])
        @inbounds fm = Fm[i, j, k, n] - 0.5f0*minmod(Fm[i, j+1, k, n] - Fm[i, j, k, n], 
                                                   Fm[i, j, k, n] - Fm[i, j-1, k, n])
        @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
    end
    return
end

function NND_z(F, Fp, Fm, NV)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG+1 || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end
    for n = 1:NV
        @inbounds fp = Fp[i, j, k-1, n] + 0.5f0*minmod(Fp[i, j, k, n] - Fp[i, j, k-1, n], 
                                                     Fp[i, j, k-1, n] - Fp[i, j, k-2, n])
        @inbounds fm = Fm[i, j, k, n] - 0.5f0*minmod(Fm[i, j, k+1, n] - Fm[i, j, k, n], 
                                                   Fm[i, j, k, n] - Fm[i, j, k-1, n])
        @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
    end
    return
end

#= 
    Advection in three directions
    UP7 + WENO7 + WENO5 + NND2
=#
function advect_x(F, ϕ, S, Fp, Fm, NV)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+NG+1 || j > Nyp+NG || k > Nzp+NG || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end

    tmp1::Float32 = 1/12f0
    tmp2::Float32 = 1/6f0
    WENOϵ1::Float64 = 1e-20
    WENOϵ2::Float32 = 1f-16

    c1::Float32 = UP7[1]
    c2::Float32 = UP7[2]
    c3::Float32 = UP7[3]
    c4::Float32 = UP7[4]
    c5::Float32 = UP7[5]
    c6::Float32 = UP7[6]
    c7::Float32 = UP7[7]

    @inbounds ss::Float32 = 2/(S[i-1, j, k] + S[i, j, k]) 

    # Jameson sensor
    @inbounds ϕx = max(ϕ[i-1, j, k], ϕ[i, j, k])

    if ϕx < hybrid_ϕ1
        for n = 1:NV
            @inbounds V1 = Fp[i-4, j, k, n]
            @inbounds V2 = Fp[i-3, j, k, n]
            @inbounds V3 = Fp[i-2, j, k, n]
            @inbounds V4 = Fp[i-1, j, k, n]
            @inbounds V5 = Fp[i,   j, k, n]
            @inbounds V6 = Fp[i+1, j, k, n]
            @inbounds V7 = Fp[i+2, j, k, n]
            
            fp = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6 + c7*V7

            @inbounds V1 = Fm[i+3, j, k, n]
            @inbounds V2 = Fm[i+2, j, k, n]
            @inbounds V3 = Fm[i+1, j, k, n]
            @inbounds V4 = Fm[i,   j, k, n]
            @inbounds V5 = Fm[i-1, j, k, n]
            @inbounds V6 = Fm[i-2, j, k, n]
            @inbounds V7 = Fm[i-3, j, k, n]

            fm = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6 + c7*V7

            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end
    elseif ϕx < hybrid_ϕ2
        for n = 1:NV
            @inbounds V1 = Fp[i-4, j, k, n]
            @inbounds V2 = Fp[i-3, j, k, n]
            @inbounds V3 = Fp[i-2, j, k, n]
            @inbounds V4 = Fp[i-1, j, k, n]
            @inbounds V5 = Fp[i,   j, k, n]
            @inbounds V6 = Fp[i+1, j, k, n]
            @inbounds V7 = Fp[i+2, j, k, n]
    
            # polinomia
            q1 = -3V1+13V2-23V3+25V4
            q2 = V2-5V3+13V4+3V5
            q3 = -V3+7V4+7V5-V6
            q4 = 3V4+13V5-5V6+V7
    
            # smoothness index
            Is1 = V1*( 547V1 - 3882V2 + 4642V3 - 1854V4) +
                  V2*(         7043V2 -17246V3 + 7042V4) +
                  V3*(                 11003V3 - 9402V4) +
                  V4*(                           2107V4)
            Is2 = V2*( 267V2 - 1642V3 + 1602V4 -  494V5) +
                  V3*(         2843V3 - 5966V4 + 1922V5) +
                  V4*(                  3443V4 - 2522V5) +
                  V5*(                            547V5)
            Is3 = V3*( 547V3 - 2522V4 + 1922V5 -  494V6) +
                  V4*(         3443V4 - 5966V5 + 1602V6) +
                  V5*(                  2843V5 - 1642V6) +
                  V6*(                            267V6)
            Is4 = V4*(2107V4 - 9402V5 + 7042V6 - 1854V7) +
                  V5*(        11003V5 -17246V6 + 4642V7) +
                  V6*(                  7043V6 - 3882V7) +
                  V7*(                            547V7)
    
            # alpha
            α1 = 1/(WENOϵ1+Is1*ss)^2
            α2 = 12/(WENOϵ1+Is2*ss)^2
            α3 = 18/(WENOϵ1+Is3*ss)^2
            α4 = 4/(WENOϵ1+Is4*ss)^2
    
            invsum = 1/(α1+α2+α3+α4)
    
            fp = invsum*(α1*q1+α2*q2+α3*q3+α4*q4)
    
            @inbounds V1 = Fm[i+3, j, k, n]
            @inbounds V2 = Fm[i+2, j, k, n]
            @inbounds V3 = Fm[i+1, j, k, n]
            @inbounds V4 = Fm[i,   j, k, n]
            @inbounds V5 = Fm[i-1, j, k, n]
            @inbounds V6 = Fm[i-2, j, k, n]
            @inbounds V7 = Fm[i-3, j, k, n]
    
            # polinomia
            q1 = -3V1+13V2-23V3+25V4
            q2 = V2-5V3+13V4+3V5
            q3 = -V3+7V4+7V5-V6
            q4 = 3V4+13V5-5V6+V7
    
            # smoothness index
            Is1 = V1*( 547V1 - 3882V2 + 4642V3 - 1854V4) +
                  V2*(         7043V2 -17246V3 + 7042V4) +
                  V3*(                 11003V3 - 9402V4) +
                  V4*(                           2107V4)
            Is2 = V2*( 267V2 - 1642V3 + 1602V4 -  494V5) +
                  V3*(         2843V3 - 5966V4 + 1922V5) +
                  V4*(                  3443V4 - 2522V5) +
                  V5*(                            547V5)
            Is3 = V3*( 547V3 - 2522V4 + 1922V5 -  494V6) +
                  V4*(         3443V4 - 5966V5 + 1602V6) +
                  V5*(                  2843V5 - 1642V6) +
                  V6*(                            267V6)
            Is4 = V4*(2107V4 - 9402V5 + 7042V6 - 1854V7) +
                  V5*(        11003V5 -17246V6 + 4642V7) +
                  V6*(                  7043V6 - 3882V7) +
                  V7*(                            547V7)
    
            # alpha
            α1 = 1/(WENOϵ1+Is1*ss)^2
            α2 = 12/(WENOϵ1+Is2*ss)^2
            α3 = 18/(WENOϵ1+Is3*ss)^2
            α4 = 4/(WENOϵ1+Is4*ss)^2
    
            invsum = 1/(α1+α2+α3+α4)
    
            fm = invsum*(α1*q1+α2*q2+α3*q3+α4*q4)
    
            @inbounds F[i-NG, j-NG, k-NG, n] = (fp + fm) * tmp1
        end
    elseif ϕx < hybrid_ϕ3
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

            s11 = 1/(WENOϵ2+s11*ss)^2
            s22 = 6/(WENOϵ2+s22*ss)^2
            s33 = 3/(WENOϵ2+s33*ss)^2

            invsum = 1/(s11+s22+s33)

            v1 = 2*V1-7*V2+11*V3
            v2 = -V2+5*V3+2*V4
            v3 = 2*V3+5*V4-V5
            fp = invsum*(s11*v1+s22*v2+s33*v3)

            @inbounds V1 = Fm[i+2, j, k, n]
            @inbounds V2 = Fm[i+1, j, k, n]
            @inbounds V3 = Fm[i,   j, k, n]
            @inbounds V4 = Fm[i-1, j, k, n]
            @inbounds V5 = Fm[i-2, j, k, n]
            # FM
            s11 = 13*(V1-2*V2+V3)^2 + 3*(V1-4*V2+3*V3)^2
            s22 = 13*(V2-2*V3+V4)^2 + 3*(V2-V4)^2
            s33 = 13*(V3-2*V4+V5)^2 + 3*(3*V3-4*V4+V5)^2

            s11 = 1/(WENOϵ2+s11*ss)^2
            s22 = 6/(WENOϵ2+s22*ss)^2
            s33 = 3/(WENOϵ2+s33*ss)^2

            invsum = 1/(s11+s22+s33)

            v1 = 2*V1-7*V2+11*V3
            v2 = -V2+5*V3+2*V4
            v3 = 2*V3+5*V4-V5
            fm = invsum*(s11*v1+s22*v2+s33*v3)
            
            @inbounds F[i-NG, j-NG, k-NG, n] = (fp + fm) * tmp2
        end
    else
        for n = 1:NV
            @inbounds fp = Fp[i-1, j, k, n] + 0.5f0*minmod(Fp[i, j, k, n] - Fp[i-1, j, k, n], 
                                                         Fp[i-1, j, k, n] - Fp[i-2, j, k, n])
            @inbounds fm = Fm[i, j, k, n] - 0.5f0*minmod(Fm[i+1, j, k, n] - Fm[i, j, k, n], 
                                                       Fm[i, j, k, n] - Fm[i-1, j, k, n])
            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end 
    end
    return
end

function advect_y(F, ϕ, S, Fp, Fm, NV)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+NG || j > Nyp+NG+1 || k > Nzp+NG || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end

    tmp1::Float32 = 1/12f0
    tmp2::Float32 = 1/6f0
    WENOϵ1::Float64 = 1e-20
    WENOϵ2::Float32 = 1f-16

    c1::Float32 = UP7[1]
    c2::Float32 = UP7[2]
    c3::Float32 = UP7[3]
    c4::Float32 = UP7[4]
    c5::Float32 = UP7[5]
    c6::Float32 = UP7[6]
    c7::Float32 = UP7[7]

    @inbounds ss::Float32 = 2/(S[i, j-1, k] + S[i, j, k]) 

    # Jameson sensor
    @inbounds ϕy = max(ϕ[i, j-1, k], ϕ[i, j, k])

    if ϕy < hybrid_ϕ1
        for n = 1:NV
            @inbounds V1 = Fp[i, j-4, k, n]
            @inbounds V2 = Fp[i, j-3, k, n]
            @inbounds V3 = Fp[i, j-2, k, n]
            @inbounds V4 = Fp[i, j-1, k, n]
            @inbounds V5 = Fp[i, j,   k, n]
            @inbounds V6 = Fp[i, j+1, k, n]
            @inbounds V7 = Fp[i, j+2, k, n]
            
            fp = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6 + c7*V7

            @inbounds V1 = Fm[i, j+3, k, n]
            @inbounds V2 = Fm[i, j+2, k, n]
            @inbounds V3 = Fm[i, j+1, k, n]
            @inbounds V4 = Fm[i, j,   k, n]
            @inbounds V5 = Fm[i, j-1, k, n]
            @inbounds V6 = Fm[i, j-2, k, n]
            @inbounds V7 = Fm[i, j-3, k, n]

            fm = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6 + c7*V7

            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end
    elseif ϕy < hybrid_ϕ2
        for n = 1:NV
            @inbounds V1 = Fp[i, j-4, k, n]
            @inbounds V2 = Fp[i, j-3, k, n]
            @inbounds V3 = Fp[i, j-2, k, n]
            @inbounds V4 = Fp[i, j-1, k, n]
            @inbounds V5 = Fp[i, j,   k, n]
            @inbounds V6 = Fp[i, j+1, k, n]
            @inbounds V7 = Fp[i, j+2, k, n]
    
            # polinomia
            q1 = -3V1+13V2-23V3+25V4
            q2 = V2-5V3+13V4+3V5
            q3 = -V3+7V4+7V5-V6
            q4 = 3V4+13V5-5V6+V7
    
            # smoothness index
            Is1 = V1*( 547V1 - 3882V2 + 4642V3 - 1854V4) +
                  V2*(         7043V2 -17246V3 + 7042V4) +
                  V3*(                 11003V3 - 9402V4) +
                  V4*(                           2107V4)
            Is2 = V2*( 267V2 - 1642V3 + 1602V4 -  494V5) +
                  V3*(         2843V3 - 5966V4 + 1922V5) +
                  V4*(                  3443V4 - 2522V5) +
                  V5*(                            547V5)
            Is3 = V3*( 547V3 - 2522V4 + 1922V5 -  494V6) +
                  V4*(         3443V4 - 5966V5 + 1602V6) +
                  V5*(                  2843V5 - 1642V6) +
                  V6*(                            267V6)
            Is4 = V4*(2107V4 - 9402V5 + 7042V6 - 1854V7) +
                  V5*(        11003V5 -17246V6 + 4642V7) +
                  V6*(                  7043V6 - 3882V7) +
                  V7*(                            547V7)
    
            # alpha
            α1 = 1/(WENOϵ1+Is1*ss)^2
            α2 = 12/(WENOϵ1+Is2*ss)^2
            α3 = 18/(WENOϵ1+Is3*ss)^2
            α4 = 4/(WENOϵ1+Is4*ss)^2
    
            invsum = 1/(α1+α2+α3+α4)
    
            fp = invsum*(α1*q1+α2*q2+α3*q3+α4*q4)
    
            @inbounds V1 = Fm[i, j+3, k, n]
            @inbounds V2 = Fm[i, j+2, k, n]
            @inbounds V3 = Fm[i, j+1, k, n]
            @inbounds V4 = Fm[i, j,   k, n]
            @inbounds V5 = Fm[i, j-1, k, n]
            @inbounds V6 = Fm[i, j-2, k, n]
            @inbounds V7 = Fm[i, j-3, k, n]
    
            # polinomia
            q1 = -3V1+13V2-23V3+25V4
            q2 = V2-5V3+13V4+3V5
            q3 = -V3+7V4+7V5-V6
            q4 = 3V4+13V5-5V6+V7
    
            # smoothness index
            Is1 = V1*( 547V1 - 3882V2 + 4642V3 - 1854V4) +
                  V2*(         7043V2 -17246V3 + 7042V4) +
                  V3*(                 11003V3 - 9402V4) +
                  V4*(                           2107V4)
            Is2 = V2*( 267V2 - 1642V3 + 1602V4 -  494V5) +
                  V3*(         2843V3 - 5966V4 + 1922V5) +
                  V4*(                  3443V4 - 2522V5) +
                  V5*(                            547V5)
            Is3 = V3*( 547V3 - 2522V4 + 1922V5 -  494V6) +
                  V4*(         3443V4 - 5966V5 + 1602V6) +
                  V5*(                  2843V5 - 1642V6) +
                  V6*(                            267V6)
            Is4 = V4*(2107V4 - 9402V5 + 7042V6 - 1854V7) +
                  V5*(        11003V5 -17246V6 + 4642V7) +
                  V6*(                  7043V6 - 3882V7) +
                  V7*(                            547V7)
    
            # alpha
            α1 = 1/(WENOϵ1+Is1*ss)^2
            α2 = 12/(WENOϵ1+Is2*ss)^2
            α3 = 18/(WENOϵ1+Is3*ss)^2
            α4 = 4/(WENOϵ1+Is4*ss)^2
    
            invsum = 1/(α1+α2+α3+α4)
    
            fm = invsum*(α1*q1+α2*q2+α3*q3+α4*q4)
    
            @inbounds F[i-NG, j-NG, k-NG, n] = (fp + fm) * tmp1
        end
    elseif ϕy < hybrid_ϕ3
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

            s11 = 1/(WENOϵ2+s11*ss)^2
            s22 = 6/(WENOϵ2+s22*ss)^2
            s33 = 3/(WENOϵ2+s33*ss)^2

            invsum = 1/(s11+s22+s33)

            v1 = 2*V1-7*V2+11*V3
            v2 = -V2+5*V3+2*V4
            v3 = 2*V3+5*V4-V5
            fp = invsum*(s11*v1+s22*v2+s33*v3)

            @inbounds V1 = Fm[i, j+2, k, n]
            @inbounds V2 = Fm[i, j+1, k, n]
            @inbounds V3 = Fm[i, j,   k, n]
            @inbounds V4 = Fm[i, j-1, k, n]
            @inbounds V5 = Fm[i, j-2, k, n]
            # FM
            s11 = 13*(V1-2*V2+V3)^2 + 3*(V1-4*V2+3*V3)^2
            s22 = 13*(V2-2*V3+V4)^2 + 3*(V2-V4)^2
            s33 = 13*(V3-2*V4+V5)^2 + 3*(3*V3-4*V4+V5)^2

            s11 = 1/(WENOϵ2+s11*ss)^2
            s22 = 6/(WENOϵ2+s22*ss)^2
            s33 = 3/(WENOϵ2+s33*ss)^2

            invsum = 1/(s11+s22+s33)

            v1 = 2*V1-7*V2+11*V3
            v2 = -V2+5*V3+2*V4
            v3 = 2*V3+5*V4-V5
            fm = invsum*(s11*v1+s22*v2+s33*v3)
            
            @inbounds F[i-NG, j-NG, k-NG, n] = (fp + fm) * tmp2
        end
    else
        for n = 1:NV
            @inbounds fp = Fp[i, j-1, k, n] + 0.5f0*minmod(Fp[i, j, k, n] - Fp[i, j-1, k, n], 
                                                         Fp[i, j-1, k, n] - Fp[i, j-2, k, n])
            @inbounds fm = Fm[i, j, k, n] - 0.5f0*minmod(Fm[i, j+1, k, n] - Fm[i, j, k, n], 
                                                       Fm[i, j, k, n] - Fm[i, j-1, k, n])
            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end 
    end
    return
end

function advect_z(F, ϕ, S, Fp, Fm, NV)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG+1 || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end

    tmp1::Float32 = 1/12f0
    tmp2::Float32 = 1/6f0
    WENOϵ1::Float64 = 1e-20
    WENOϵ2::Float32 = 1f-16

    c1::Float32 = UP7[1]
    c2::Float32 = UP7[2]
    c3::Float32 = UP7[3]
    c4::Float32 = UP7[4]
    c5::Float32 = UP7[5]
    c6::Float32 = UP7[6]
    c7::Float32 = UP7[7]

    @inbounds ss::Float32 = 2/(S[i, j, k-1] + S[i, j, k]) 

    # Jameson sensor
    @inbounds ϕz = max(ϕ[i, j, k-1], ϕ[i, j, k])

    if ϕz < hybrid_ϕ1
        for n = 1:NV
            @inbounds V1 = Fp[i, j, k-4, n]
            @inbounds V2 = Fp[i, j, k-3, n]
            @inbounds V3 = Fp[i, j, k-2, n]
            @inbounds V4 = Fp[i, j, k-1, n]
            @inbounds V5 = Fp[i, j, k,   n]
            @inbounds V6 = Fp[i, j, k+1, n]
            @inbounds V7 = Fp[i, j, k+2, n]
            
            fp = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6 + c7*V7

            @inbounds V1 = Fm[i, j, k+3, n]
            @inbounds V2 = Fm[i, j, k+2, n]
            @inbounds V3 = Fm[i, j, k+1, n]
            @inbounds V4 = Fm[i, j, k,   n]
            @inbounds V5 = Fm[i, j, k-1, n]
            @inbounds V6 = Fm[i, j, k-2, n]
            @inbounds V7 = Fm[i, j, k-3, n]

            fm = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6 + c7*V7

            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end
    elseif ϕz < hybrid_ϕ2
        for n = 1:NV
            @inbounds V1 = Fp[i, j, k-4, n]
            @inbounds V2 = Fp[i, j, k-3, n]
            @inbounds V3 = Fp[i, j, k-2, n]
            @inbounds V4 = Fp[i, j, k-1, n]
            @inbounds V5 = Fp[i, j, k,   n]
            @inbounds V6 = Fp[i, j, k+1, n]
            @inbounds V7 = Fp[i, j, k+2, n]
    
            # polinomia
            q1 = -3V1+13V2-23V3+25V4
            q2 = V2-5V3+13V4+3V5
            q3 = -V3+7V4+7V5-V6
            q4 = 3V4+13V5-5V6+V7
    
            # smoothness index
            Is1 = V1*( 547V1 - 3882V2 + 4642V3 - 1854V4) +
                  V2*(         7043V2 -17246V3 + 7042V4) +
                  V3*(                 11003V3 - 9402V4) +
                  V4*(                           2107V4)
            Is2 = V2*( 267V2 - 1642V3 + 1602V4 -  494V5) +
                  V3*(         2843V3 - 5966V4 + 1922V5) +
                  V4*(                  3443V4 - 2522V5) +
                  V5*(                            547V5)
            Is3 = V3*( 547V3 - 2522V4 + 1922V5 -  494V6) +
                  V4*(         3443V4 - 5966V5 + 1602V6) +
                  V5*(                  2843V5 - 1642V6) +
                  V6*(                            267V6)
            Is4 = V4*(2107V4 - 9402V5 + 7042V6 - 1854V7) +
                  V5*(        11003V5 -17246V6 + 4642V7) +
                  V6*(                  7043V6 - 3882V7) +
                  V7*(                            547V7)
    
            # alpha
            α1 = 1/(WENOϵ1+Is1*ss)^2
            α2 = 12/(WENOϵ1+Is2*ss)^2
            α3 = 18/(WENOϵ1+Is3*ss)^2
            α4 = 4/(WENOϵ1+Is4*ss)^2
    
            invsum = 1/(α1+α2+α3+α4)
    
            fp = invsum*(α1*q1+α2*q2+α3*q3+α4*q4)
    
            @inbounds V1 = Fm[i, j, k+3, n]
            @inbounds V2 = Fm[i, j, k+2, n]
            @inbounds V3 = Fm[i, j, k+1, n]
            @inbounds V4 = Fm[i, j, k,   n]
            @inbounds V5 = Fm[i, j, k-1, n]
            @inbounds V6 = Fm[i, j, k-2, n]
            @inbounds V7 = Fm[i, j, k-3, n]
    
            # polinomia
            q1 = -3V1+13V2-23V3+25V4
            q2 = V2-5V3+13V4+3V5
            q3 = -V3+7V4+7V5-V6
            q4 = 3V4+13V5-5V6+V7
    
            # smoothness index
            Is1 = V1*( 547V1 - 3882V2 + 4642V3 - 1854V4) +
                  V2*(         7043V2 -17246V3 + 7042V4) +
                  V3*(                 11003V3 - 9402V4) +
                  V4*(                           2107V4)
            Is2 = V2*( 267V2 - 1642V3 + 1602V4 -  494V5) +
                  V3*(         2843V3 - 5966V4 + 1922V5) +
                  V4*(                  3443V4 - 2522V5) +
                  V5*(                            547V5)
            Is3 = V3*( 547V3 - 2522V4 + 1922V5 -  494V6) +
                  V4*(         3443V4 - 5966V5 + 1602V6) +
                  V5*(                  2843V5 - 1642V6) +
                  V6*(                            267V6)
            Is4 = V4*(2107V4 - 9402V5 + 7042V6 - 1854V7) +
                  V5*(        11003V5 -17246V6 + 4642V7) +
                  V6*(                  7043V6 - 3882V7) +
                  V7*(                            547V7)
    
            # alpha
            α1 = 1/(WENOϵ1+Is1*ss)^2
            α2 = 12/(WENOϵ1+Is2*ss)^2
            α3 = 18/(WENOϵ1+Is3*ss)^2
            α4 = 4/(WENOϵ1+Is4*ss)^2
    
            invsum = 1/(α1+α2+α3+α4)
    
            fm = invsum*(α1*q1+α2*q2+α3*q3+α4*q4)
    
            @inbounds F[i-NG, j-NG, k-NG, n] = (fp + fm) * tmp1
        end
    elseif ϕz < hybrid_ϕ3
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

            s11 = 1/(WENOϵ2+s11*ss)^2
            s22 = 6/(WENOϵ2+s22*ss)^2
            s33 = 3/(WENOϵ2+s33*ss)^2

            invsum = 1/(s11+s22+s33)

            v1 = 2*V1-7*V2+11*V3
            v2 = -V2+5*V3+2*V4
            v3 = 2*V3+5*V4-V5
            fp = invsum*(s11*v1+s22*v2+s33*v3)

            @inbounds V1 = Fm[i, j, k+2, n]
            @inbounds V2 = Fm[i, j, k+1, n]
            @inbounds V3 = Fm[i, j, k,   n]
            @inbounds V4 = Fm[i, j, k-1, n]
            @inbounds V5 = Fm[i, j, k-2, n]
            # FM
            s11 = 13*(V1-2*V2+V3)^2 + 3*(V1-4*V2+3*V3)^2
            s22 = 13*(V2-2*V3+V4)^2 + 3*(V2-V4)^2
            s33 = 13*(V3-2*V4+V5)^2 + 3*(3*V3-4*V4+V5)^2

            s11 = 1/(WENOϵ2+s11*ss)^2
            s22 = 6/(WENOϵ2+s22*ss)^2
            s33 = 3/(WENOϵ2+s33*ss)^2

            invsum = 1/(s11+s22+s33)

            v1 = 2*V1-7*V2+11*V3
            v2 = -V2+5*V3+2*V4
            v3 = 2*V3+5*V4-V5
            fm = invsum*(s11*v1+s22*v2+s33*v3)
            
            @inbounds F[i-NG, j-NG, k-NG, n] = (fp + fm) * tmp2
        end
    else
        for n = 1:NV
            @inbounds fp = Fp[i, j, k-1, n] + 0.5f0*minmod(Fp[i, j, k, n] - Fp[i, j, k-1, n], 
                                                         Fp[i, j, k-1, n] - Fp[i, j, k-2, n])
            @inbounds fm = Fm[i, j, k, n] - 0.5f0*minmod(Fm[i, j, k+1, n] - Fm[i, j, k, n], 
                                                       Fm[i, j, k, n] - Fm[i, j, k-1, n])
            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end
    end
    return
end

function advect_xc(F, ϕ, S, Fp, Fm, Q, Ax, Ay, Az)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+NG+1 || j > Nyp+NG || k > Nzp+NG || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end

    tmp1::Float32 = 1/12f0
    tmp2::Float32 = 1/6f0

    WENOϵ1::Float64 = 1e-20
    WENOϵ2::Float32 = 1f-16

    c1::Float32 = UP7[1]
    c2::Float32 = UP7[2]
    c3::Float32 = UP7[3]
    c4::Float32 = UP7[4]
    c5::Float32 = UP7[5]
    c6::Float32 = UP7[6]
    c7::Float32 = UP7[7]

    # Jameson sensor
    @inbounds ϕx = max(ϕ[i-3, j, k], 
                       ϕ[i-2, j, k], 
                       ϕ[i-1, j, k], 
                       ϕ[i,   j, k], 
                       ϕ[i+1, j, k], 
                       ϕ[i+2, j, k])

    if ϕx < hybrid_ϕ1
        for n = 1:Ncons
            @inbounds V1 = Fp[i-4, j, k, n]
            @inbounds V2 = Fp[i-3, j, k, n]
            @inbounds V3 = Fp[i-2, j, k, n]
            @inbounds V4 = Fp[i-1, j, k, n]
            @inbounds V5 = Fp[i,   j, k, n]
            @inbounds V6 = Fp[i+1, j, k, n]
            @inbounds V7 = Fp[i+2, j, k, n]
            
            fp = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6 + c7*V7

            @inbounds V1 = Fm[i+3, j, k, n]
            @inbounds V2 = Fm[i+2, j, k, n]
            @inbounds V3 = Fm[i+1, j, k, n]
            @inbounds V4 = Fm[i,   j, k, n]
            @inbounds V5 = Fm[i-1, j, k, n]
            @inbounds V6 = Fm[i-2, j, k, n]
            @inbounds V7 = Fm[i-3, j, k, n]

            fm = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6 + c7*V7

            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end
    else
        @inbounds s0::Float32 = 2/(S[i-1, j, k] + S[i, j, k]) 

        # average
        @inbounds u = 0.5f0 * (Q[i, j, k, 2] + Q[i-1, j, k, 2])
        @inbounds v = 0.5f0 * (Q[i, j, k, 3] + Q[i-1, j, k, 3])
        @inbounds w = 0.5f0 * (Q[i, j, k, 4] + Q[i-1, j, k, 4])
        @inbounds T = 0.5f0 * (Q[i, j, k, 6] + Q[i-1, j, k, 6])
    
        @inbounds A1 = (Ax[i, j, k] + Ax[i-1, j, k]) * 0.5f0
        @inbounds A2 = (Ay[i, j, k] + Ay[i-1, j, k]) * 0.5f0
        @inbounds A3 = (Az[i, j, k] + Az[i-1, j, k]) * 0.5f0
    
        nx = A1*s0
        ny = A2*s0
        nz = A3*s0
    
        if abs(nz) <= abs(ny)
            ss = 1/sqrt(nx*nx+ny*ny) 
            lx = -ny*ss
            ly = nx*ss
            lz = 0.f0 
        else 
            ss = 1/sqrt(nx*nx+nz*nz)
            lx = -nz*ss
            ly = 0.f0   
            lz = nx*ss
        end 
        mx = ny*lz-nz*ly 
        my = nz*lx-nx*lz
        mz = nx*ly-ny*lx
    
        qn = u*nx+v*ny+w*nz 
        ql = u*lx+v*ly+w*lz 
        qm = u*mx+v*my+w*mz 
        q2 = 0.5f0*(u^2+v^2+w^2)
        c = sqrt(γ*Rg*T)
        c2 = 1/(2*c)
        K = (γ-1)/c^2 
        K2 = K * 0.5f0
        H = 1/K+q2
    
        L = MMatrix{Ncons, Ncons, Float32, Ncons*Ncons}(undef)
        R = MMatrix{Ncons, Ncons, Float32, Ncons*Ncons}(undef)
        flux = MVector{Ncons, Float32}(undef)
    
        L[1,1]=K2*q2+qn*c2;     L[1,2]=-(K2*u+nx*c2);     L[1,3]=-(K2*v+ny*c2);     L[1,4]=-(K2*w+nz*c2);     L[1,5]=K2
        L[2,1]=1-K*q2;          L[2,2]=K*u;               L[2,3]=K*v;               L[2,4]=K*w;               L[2,5]=-K 
        L[3,1]=K2*q2-qn*c2;     L[3,2]=-(K2*u-nx*c2);     L[3,3]=-(K2*v-ny*c2);     L[3,4]=-(K2*w-nz*c2);     L[3,5]=K2
        L[4,1]=-ql;             L[4,2]=lx;                L[4,3]=ly;                L[4,4]=lz;                L[4,5]=0.f0
        L[5,1]=-qm;             L[5,2]=mx;                L[5,3]=my;                L[5,4]=mz;                L[5,5]=0.f0
    
        R[1,1]=1.f0;      R[1,2]=1.f0;       R[1,3]=1.f0;      R[1,4]=0.f0;    R[1,5]=0.f0
        R[2,1]=u-c*nx;    R[2,2]=u;          R[2,3]=u+c*nx;    R[2,4]=lx;      R[2,5]=mx
        R[3,1]=v-c*ny;    R[3,2]=v;          R[3,3]=v+c*ny;    R[3,4]=ly;      R[3,5]=my		
        R[4,1]=w-c*nz;    R[4,2]=w;          R[4,3]=w+c*nz;    R[4,4]=lz;      R[4,5]=mz		
        R[5,1]=H-qn*c;    R[5,2]=q2;         R[5,3]=H+qn*c;    R[5,4]=ql;      R[5,5]=qm

        if ϕx < hybrid_ϕ2
            Fpc = MVector{7, Float32}(undef)
            Fmc = MVector{7, Float32}(undef)

            for n = 1:Ncons
                Fpc .= 0.f0
                Fmc .= 0.f0
                for m = 1:Ncons
                    ll = L[n, m]
                    @inbounds Fpc[1] += Fp[i-4, j, k, m] * ll
                    @inbounds Fpc[2] += Fp[i-3, j, k, m] * ll
                    @inbounds Fpc[3] += Fp[i-2, j, k, m] * ll
                    @inbounds Fpc[4] += Fp[i-1, j, k, m] * ll
                    @inbounds Fpc[5] += Fp[i,   j, k, m] * ll
                    @inbounds Fpc[6] += Fp[i+1, j, k, m] * ll
                    @inbounds Fpc[7] += Fp[i+2, j, k, m] * ll
        
                    @inbounds Fmc[1] += Fm[i-3, j, k, m] * ll
                    @inbounds Fmc[2] += Fm[i-2, j, k, m] * ll
                    @inbounds Fmc[3] += Fm[i-1, j, k, m] * ll
                    @inbounds Fmc[4] += Fm[i,   j, k, m] * ll
                    @inbounds Fmc[5] += Fm[i+1, j, k, m] * ll
                    @inbounds Fmc[6] += Fm[i+2, j, k, m] * ll
                    @inbounds Fmc[7] += Fm[i+3, j, k, m] * ll
                end

                @inbounds V1 = Fpc[1]
                @inbounds V2 = Fpc[2]
                @inbounds V3 = Fpc[3]
                @inbounds V4 = Fpc[4]
                @inbounds V5 = Fpc[5]
                @inbounds V6 = Fpc[6]
                @inbounds V7 = Fpc[7]
        
                # polinomia
                q1 = -3V1+13V2-23V3+25V4
                q2 = V2-5V3+13V4+3V5
                q3 = -V3+7V4+7V5-V6
                q4 = 3V4+13V5-5V6+V7
        
                # smoothness index
                Is1 = V1*( 547V1 - 3882V2 + 4642V3 - 1854V4) +
                    V2*(         7043V2 -17246V3 + 7042V4) +
                    V3*(                 11003V3 - 9402V4) +
                    V4*(                           2107V4)
                Is2 = V2*( 267V2 - 1642V3 + 1602V4 -  494V5) +
                    V3*(         2843V3 - 5966V4 + 1922V5) +
                    V4*(                  3443V4 - 2522V5) +
                    V5*(                            547V5)
                Is3 = V3*( 547V3 - 2522V4 + 1922V5 -  494V6) +
                    V4*(         3443V4 - 5966V5 + 1602V6) +
                    V5*(                  2843V5 - 1642V6) +
                    V6*(                            267V6)
                Is4 = V4*(2107V4 - 9402V5 + 7042V6 - 1854V7) +
                    V5*(        11003V5 -17246V6 + 4642V7) +
                    V6*(                  7043V6 - 3882V7) +
                    V7*(                            547V7)
        
                # alpha
                α1 = 1/(WENOϵ1+Is1*s0)^2
                α2 = 12/(WENOϵ1+Is2*s0)^2
                α3 = 18/(WENOϵ1+Is3*s0)^2
                α4 = 4/(WENOϵ1+Is4*s0)^2
        
                invsum = 1/(α1+α2+α3+α4)
        
                fp = invsum*(α1*q1+α2*q2+α3*q3+α4*q4)
        
                @inbounds V1 = Fmc[1]
                @inbounds V2 = Fmc[2]
                @inbounds V3 = Fmc[3]
                @inbounds V4 = Fmc[4]
                @inbounds V5 = Fmc[5]
                @inbounds V6 = Fmc[6]
                @inbounds V7 = Fmc[7]
        
                # polinomia
                q1 = -3V1+13V2-23V3+25V4
                q2 = V2-5V3+13V4+3V5
                q3 = -V3+7V4+7V5-V6
                q4 = 3V4+13V5-5V6+V7
        
                # smoothness index
                Is1 = V1*( 547V1 - 3882V2 + 4642V3 - 1854V4) +
                    V2*(         7043V2 -17246V3 + 7042V4) +
                    V3*(                 11003V3 - 9402V4) +
                    V4*(                           2107V4)
                Is2 = V2*( 267V2 - 1642V3 + 1602V4 -  494V5) +
                    V3*(         2843V3 - 5966V4 + 1922V5) +
                    V4*(                  3443V4 - 2522V5) +
                    V5*(                            547V5)
                Is3 = V3*( 547V3 - 2522V4 + 1922V5 -  494V6) +
                    V4*(         3443V4 - 5966V5 + 1602V6) +
                    V5*(                  2843V5 - 1642V6) +
                    V6*(                            267V6)
                Is4 = V4*(2107V4 - 9402V5 + 7042V6 - 1854V7) +
                    V5*(        11003V5 -17246V6 + 4642V7) +
                    V6*(                  7043V6 - 3882V7) +
                    V7*(                            547V7)
        
                # alpha
                α1 = 1/(WENOϵ1+Is1*s0)^2
                α2 = 12/(WENOϵ1+Is2*s0)^2
                α3 = 18/(WENOϵ1+Is3*s0)^2
                α4 = 4/(WENOϵ1+Is4*s0)^2
        
                invsum = 1/(α1+α2+α3+α4)
        
                fm = invsum*(α1*q1+α2*q2+α3*q3+α4*q4)
        
                @inbounds flux[n] = (fp + fm) * tmp1
            end

            for n = 1:Ncons
                @inbounds F[i-NG, j-NG, k-NG, n] = flux[1] * R[n, 1] + flux[2] * R[n, 2] + 
                                                   flux[3] * R[n, 3] + flux[4] * R[n, 4] + 
                                                   flux[5] * R[n, 5]
            end
        elseif ϕx < hybrid_ϕ3
            Fpc = MVector{5, Float32}(undef)
            Fmc = MVector{5, Float32}(undef)

            for n = 1:Ncons
                Fpc .= 0.f0
                Fmc .= 0.f0
                for m = 1:Ncons
                    ll = L[n, m]
                    @inbounds Fpc[1] += Fp[i-3, j, k, m] * ll
                    @inbounds Fpc[2] += Fp[i-2, j, k, m] * ll
                    @inbounds Fpc[3] += Fp[i-1, j, k, m] * ll
                    @inbounds Fpc[4] += Fp[i,   j, k, m] * ll
                    @inbounds Fpc[5] += Fp[i+1, j, k, m] * ll
        
                    @inbounds Fmc[1] += Fm[i-2, j, k, m] * ll
                    @inbounds Fmc[2] += Fm[i-1, j, k, m] * ll
                    @inbounds Fmc[3] += Fm[i,   j, k, m] * ll
                    @inbounds Fmc[4] += Fm[i+1, j, k, m] * ll
                    @inbounds Fmc[5] += Fm[i+2, j, k, m] * ll
                end

                @inbounds V1 = Fpc[1]
                @inbounds V2 = Fpc[2]
                @inbounds V3 = Fpc[3]
                @inbounds V4 = Fpc[4]
                @inbounds V5 = Fpc[5]
                # FP
                s11 = 13*(V1-2*V2+V3)^2 + 3*(V1-4*V2+3*V3)^2
                s22 = 13*(V2-2*V3+V4)^2 + 3*(V2-V4)^2
                s33 = 13*(V3-2*V4+V5)^2 + 3*(3*V3-4*V4+V5)^2

                s11 = 1/(WENOϵ2+s11*s0)^2
                s22 = 6/(WENOϵ2+s22*s0)^2
                s33 = 3/(WENOϵ2+s33*s0)^2

                invsum = 1/(s11+s22+s33)

                v1 = 2*V1-7*V2+11*V3
                v2 = -V2+5*V3+2*V4
                v3 = 2*V3+5*V4-V5
                fp = invsum*(s11*v1+s22*v2+s33*v3)

                @inbounds V1 = Fmc[1]
                @inbounds V2 = Fmc[2]
                @inbounds V3 = Fmc[3]
                @inbounds V4 = Fmc[4]
                @inbounds V5 = Fmc[5]
                # FM
                s11 = 13*(V1-2*V2+V3)^2 + 3*(V1-4*V2+3*V3)^2
                s22 = 13*(V2-2*V3+V4)^2 + 3*(V2-V4)^2
                s33 = 13*(V3-2*V4+V5)^2 + 3*(3*V3-4*V4+V5)^2

                s11 = 1/(WENOϵ2+s11*s0)^2
                s22 = 6/(WENOϵ2+s22*s0)^2
                s33 = 3/(WENOϵ2+s33*s0)^2

                invsum = 1/(s11+s22+s33)

                v1 = 2*V1-7*V2+11*V3
                v2 = -V2+5*V3+2*V4
                v3 = 2*V3+5*V4-V5
                fm = invsum*(s11*v1+s22*v2+s33*v3)
                
                @inbounds flux[n] = (fp + fm) * tmp2
            end
            for n = 1:Ncons
                @inbounds F[i-NG, j-NG, k-NG, n] = flux[1] * R[n, 1] + flux[2] * R[n, 2] + flux[3] * R[n, 3] + flux[4] * R[n, 4] + flux[5] * R[n, 5]
            end
        else
            for n = 1:Ncons
                @inbounds fp = Fp[i-1, j, k, n] + 0.5f0*minmod(Fp[i, j, k, n] - Fp[i-1, j, k, n], 
                                                            Fp[i-1, j, k, n] - Fp[i-2, j, k, n])
                @inbounds fm = Fm[i, j, k, n] - 0.5f0*minmod(Fm[i+1, j, k, n] - Fm[i, j, k, n], 
                                                            Fm[i, j, k, n] - Fm[i-1, j, k, n])

                @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
            end 
        end
    end
    return
end

function advect_yc(F, ϕ, S, Fp, Fm, Q, Ax, Ay, Az)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+NG || j > Nyp+NG+1 || k > Nzp+NG || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end

    tmp1::Float32 = 1/12f0
    tmp2::Float32 = 1/6f0

    WENOϵ1::Float64 = 1e-20
    WENOϵ2::Float32 = 1f-16

    c1::Float32 = UP7[1]
    c2::Float32 = UP7[2]
    c3::Float32 = UP7[3]
    c4::Float32 = UP7[4]
    c5::Float32 = UP7[5]
    c6::Float32 = UP7[6]
    c7::Float32 = UP7[7]

    # Jameson sensor
    @inbounds ϕx = max(ϕ[i, j-3, k], 
                       ϕ[i, j-2, k], 
                       ϕ[i, j-1, k], 
                       ϕ[i, j,   k], 
                       ϕ[i, j+1, k], 
                       ϕ[i, j+2, k])

    if ϕx < hybrid_ϕ1
        for n = 1:Ncons
            @inbounds V1 = Fp[i, j-4, k, n]
            @inbounds V2 = Fp[i, j-3, k, n]
            @inbounds V3 = Fp[i, j-2, k, n]
            @inbounds V4 = Fp[i, j-1, k, n]
            @inbounds V5 = Fp[i, j,   k, n]
            @inbounds V6 = Fp[i, j+1, k, n]
            @inbounds V7 = Fp[i, j+2, k, n]
            
            fp = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6 + c7*V7

            @inbounds V1 = Fm[i, j+3, k, n]
            @inbounds V2 = Fm[i, j+2, k, n]
            @inbounds V3 = Fm[i, j+1, k, n]
            @inbounds V4 = Fm[i, j,   k, n]
            @inbounds V5 = Fm[i, j-1, k, n]
            @inbounds V6 = Fm[i, j-2, k, n]
            @inbounds V7 = Fm[i, j-3, k, n]

            fm = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6 + c7*V7

            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end
    else
        @inbounds s0::Float32 = 2/(S[i, j-1, k] + S[i, j, k]) 

        # average
        @inbounds u = 0.5f0 * (Q[i, j, k, 2] + Q[i, j-1, k, 2])
        @inbounds v = 0.5f0 * (Q[i, j, k, 3] + Q[i, j-1, k, 3])
        @inbounds w = 0.5f0 * (Q[i, j, k, 4] + Q[i, j-1, k, 4])
        @inbounds T = 0.5f0 * (Q[i, j, k, 6] + Q[i, j-1, k, 6])
    
        @inbounds A1 = (Ax[i, j, k] + Ax[i, j-1, k]) * 0.5f0
        @inbounds A2 = (Ay[i, j, k] + Ay[i, j-1, k]) * 0.5f0
        @inbounds A3 = (Az[i, j, k] + Az[i, j-1, k]) * 0.5f0
    
        nx = A1*s0
        ny = A2*s0
        nz = A3*s0
    
        if abs(nz) <= abs(ny)
            ss = 1/sqrt(nx*nx+ny*ny) 
            lx = -ny*ss
            ly = nx*ss
            lz = 0.f0 
        else 
            ss = 1/sqrt(nx*nx+nz*nz)
            lx = -nz*ss
            ly = 0.f0   
            lz = nx*ss
        end 
        mx = ny*lz-nz*ly 
        my = nz*lx-nx*lz
        mz = nx*ly-ny*lx
    
        qn = u*nx+v*ny+w*nz 
        ql = u*lx+v*ly+w*lz 
        qm = u*mx+v*my+w*mz 
        q2 = 0.5f0*(u^2+v^2+w^2)
        c = sqrt(γ*Rg*T)
        c2 = 1/(2*c)
        K = (γ-1)/c^2 
        K2 = K * 0.5f0
        H = 1/K+q2
    
        L = MMatrix{Ncons, Ncons, Float32, Ncons*Ncons}(undef)
        R = MMatrix{Ncons, Ncons, Float32, Ncons*Ncons}(undef)
        flux = MVector{Ncons, Float32}(undef)
    
        L[1,1]=K2*q2+qn*c2;     L[1,2]=-(K2*u+nx*c2);     L[1,3]=-(K2*v+ny*c2);     L[1,4]=-(K2*w+nz*c2);     L[1,5]=K2
        L[2,1]=1-K*q2;          L[2,2]=K*u;               L[2,3]=K*v;               L[2,4]=K*w;               L[2,5]=-K 
        L[3,1]=K2*q2-qn*c2;     L[3,2]=-(K2*u-nx*c2);     L[3,3]=-(K2*v-ny*c2);     L[3,4]=-(K2*w-nz*c2);     L[3,5]=K2
        L[4,1]=-ql;             L[4,2]=lx;                L[4,3]=ly;                L[4,4]=lz;                L[4,5]=0.f0
        L[5,1]=-qm;             L[5,2]=mx;                L[5,3]=my;                L[5,4]=mz;                L[5,5]=0.f0
    
        R[1,1]=1.f0;      R[1,2]=1.f0;       R[1,3]=1.f0;      R[1,4]=0.f0;    R[1,5]=0.f0
        R[2,1]=u-c*nx;    R[2,2]=u;          R[2,3]=u+c*nx;    R[2,4]=lx;      R[2,5]=mx
        R[3,1]=v-c*ny;    R[3,2]=v;          R[3,3]=v+c*ny;    R[3,4]=ly;      R[3,5]=my		
        R[4,1]=w-c*nz;    R[4,2]=w;          R[4,3]=w+c*nz;    R[4,4]=lz;      R[4,5]=mz		
        R[5,1]=H-qn*c;    R[5,2]=q2;         R[5,3]=H+qn*c;    R[5,4]=ql;      R[5,5]=qm

        if ϕx < hybrid_ϕ2
            Fpc = MVector{7, Float32}(undef)
            Fmc = MVector{7, Float32}(undef)

            for n = 1:Ncons
                Fpc .= 0.f0
                Fmc .= 0.f0
                for m = 1:Ncons
                    ll = L[n, m]
                    @inbounds Fpc[1] += Fp[i, j-4, k, m] * ll
                    @inbounds Fpc[2] += Fp[i, j-3, k, m] * ll
                    @inbounds Fpc[3] += Fp[i, j-2, k, m] * ll
                    @inbounds Fpc[4] += Fp[i, j-1, k, m] * ll
                    @inbounds Fpc[5] += Fp[i, j,   k, m] * ll
                    @inbounds Fpc[6] += Fp[i, j+1, k, m] * ll
                    @inbounds Fpc[7] += Fp[i, j+2, k, m] * ll
        
                    @inbounds Fmc[1] += Fm[i, j-3, k, m] * ll
                    @inbounds Fmc[2] += Fm[i, j-2, k, m] * ll
                    @inbounds Fmc[3] += Fm[i, j-1, k, m] * ll
                    @inbounds Fmc[4] += Fm[i, j,   k, m] * ll
                    @inbounds Fmc[5] += Fm[i, j+1, k, m] * ll
                    @inbounds Fmc[6] += Fm[i, j+2, k, m] * ll
                    @inbounds Fmc[7] += Fm[i, j+3, k, m] * ll
                end

                @inbounds V1 = Fpc[1]
                @inbounds V2 = Fpc[2]
                @inbounds V3 = Fpc[3]
                @inbounds V4 = Fpc[4]
                @inbounds V5 = Fpc[5]
                @inbounds V6 = Fpc[6]
                @inbounds V7 = Fpc[7]
        
                # polinomia
                q1 = -3V1+13V2-23V3+25V4
                q2 = V2-5V3+13V4+3V5
                q3 = -V3+7V4+7V5-V6
                q4 = 3V4+13V5-5V6+V7
        
                # smoothness index
                Is1 = V1*( 547V1 - 3882V2 + 4642V3 - 1854V4) +
                    V2*(         7043V2 -17246V3 + 7042V4) +
                    V3*(                 11003V3 - 9402V4) +
                    V4*(                           2107V4)
                Is2 = V2*( 267V2 - 1642V3 + 1602V4 -  494V5) +
                    V3*(         2843V3 - 5966V4 + 1922V5) +
                    V4*(                  3443V4 - 2522V5) +
                    V5*(                            547V5)
                Is3 = V3*( 547V3 - 2522V4 + 1922V5 -  494V6) +
                    V4*(         3443V4 - 5966V5 + 1602V6) +
                    V5*(                  2843V5 - 1642V6) +
                    V6*(                            267V6)
                Is4 = V4*(2107V4 - 9402V5 + 7042V6 - 1854V7) +
                    V5*(        11003V5 -17246V6 + 4642V7) +
                    V6*(                  7043V6 - 3882V7) +
                    V7*(                            547V7)
        
                # alpha
                α1 = 1/(WENOϵ1+Is1*s0)^2
                α2 = 12/(WENOϵ1+Is2*s0)^2
                α3 = 18/(WENOϵ1+Is3*s0)^2
                α4 = 4/(WENOϵ1+Is4*s0)^2
        
                invsum = 1/(α1+α2+α3+α4)
        
                fp = invsum*(α1*q1+α2*q2+α3*q3+α4*q4)
        
                @inbounds V1 = Fmc[1]
                @inbounds V2 = Fmc[2]
                @inbounds V3 = Fmc[3]
                @inbounds V4 = Fmc[4]
                @inbounds V5 = Fmc[5]
                @inbounds V6 = Fmc[6]
                @inbounds V7 = Fmc[7]
        
                # polinomia
                q1 = -3V1+13V2-23V3+25V4
                q2 = V2-5V3+13V4+3V5
                q3 = -V3+7V4+7V5-V6
                q4 = 3V4+13V5-5V6+V7
        
                # smoothness index
                Is1 = V1*( 547V1 - 3882V2 + 4642V3 - 1854V4) +
                    V2*(         7043V2 -17246V3 + 7042V4) +
                    V3*(                 11003V3 - 9402V4) +
                    V4*(                           2107V4)
                Is2 = V2*( 267V2 - 1642V3 + 1602V4 -  494V5) +
                    V3*(         2843V3 - 5966V4 + 1922V5) +
                    V4*(                  3443V4 - 2522V5) +
                    V5*(                            547V5)
                Is3 = V3*( 547V3 - 2522V4 + 1922V5 -  494V6) +
                    V4*(         3443V4 - 5966V5 + 1602V6) +
                    V5*(                  2843V5 - 1642V6) +
                    V6*(                            267V6)
                Is4 = V4*(2107V4 - 9402V5 + 7042V6 - 1854V7) +
                    V5*(        11003V5 -17246V6 + 4642V7) +
                    V6*(                  7043V6 - 3882V7) +
                    V7*(                            547V7)
        
                # alpha
                α1 = 1/(WENOϵ1+Is1*s0)^2
                α2 = 12/(WENOϵ1+Is2*s0)^2
                α3 = 18/(WENOϵ1+Is3*s0)^2
                α4 = 4/(WENOϵ1+Is4*s0)^2
        
                invsum = 1/(α1+α2+α3+α4)
        
                fm = invsum*(α1*q1+α2*q2+α3*q3+α4*q4)
        
                @inbounds flux[n] = (fp + fm) * tmp1
            end

            for n = 1:Ncons
                @inbounds F[i-NG, j-NG, k-NG, n] = flux[1] * R[n, 1] + flux[2] * R[n, 2] + 
                                                   flux[3] * R[n, 3] + flux[4] * R[n, 4] + 
                                                   flux[5] * R[n, 5]
            end
        elseif ϕx < hybrid_ϕ3
            Fpc = MVector{5, Float32}(undef)
            Fmc = MVector{5, Float32}(undef)

            for n = 1:Ncons
                Fpc .= 0.f0
                Fmc .= 0.f0
                for m = 1:Ncons
                    ll = L[n, m]
                    @inbounds Fpc[1] += Fp[i, j-3, k, m] * ll
                    @inbounds Fpc[2] += Fp[i, j-2, k, m] * ll
                    @inbounds Fpc[3] += Fp[i, j-1, k, m] * ll
                    @inbounds Fpc[4] += Fp[i, j,   k, m] * ll
                    @inbounds Fpc[5] += Fp[i, j+1, k, m] * ll
        
                    @inbounds Fmc[1] += Fm[i, j-2, k, m] * ll
                    @inbounds Fmc[2] += Fm[i, j-1, k, m] * ll
                    @inbounds Fmc[3] += Fm[i, j,   k, m] * ll
                    @inbounds Fmc[4] += Fm[i, j+1, k, m] * ll
                    @inbounds Fmc[5] += Fm[i, j+2, k, m] * ll
                end

                @inbounds V1 = Fpc[1]
                @inbounds V2 = Fpc[2]
                @inbounds V3 = Fpc[3]
                @inbounds V4 = Fpc[4]
                @inbounds V5 = Fpc[5]
                # FP
                s11 = 13*(V1-2*V2+V3)^2 + 3*(V1-4*V2+3*V3)^2
                s22 = 13*(V2-2*V3+V4)^2 + 3*(V2-V4)^2
                s33 = 13*(V3-2*V4+V5)^2 + 3*(3*V3-4*V4+V5)^2

                s11 = 1/(WENOϵ2+s11*s0)^2
                s22 = 6/(WENOϵ2+s22*s0)^2
                s33 = 3/(WENOϵ2+s33*s0)^2

                invsum = 1/(s11+s22+s33)

                v1 = 2*V1-7*V2+11*V3
                v2 = -V2+5*V3+2*V4
                v3 = 2*V3+5*V4-V5
                fp = invsum*(s11*v1+s22*v2+s33*v3)

                @inbounds V1 = Fmc[1]
                @inbounds V2 = Fmc[2]
                @inbounds V3 = Fmc[3]
                @inbounds V4 = Fmc[4]
                @inbounds V5 = Fmc[5]
                # FM
                s11 = 13*(V1-2*V2+V3)^2 + 3*(V1-4*V2+3*V3)^2
                s22 = 13*(V2-2*V3+V4)^2 + 3*(V2-V4)^2
                s33 = 13*(V3-2*V4+V5)^2 + 3*(3*V3-4*V4+V5)^2

                s11 = 1/(WENOϵ2+s11*s0)^2
                s22 = 6/(WENOϵ2+s22*s0)^2
                s33 = 3/(WENOϵ2+s33*s0)^2

                invsum = 1/(s11+s22+s33)

                v1 = 2*V1-7*V2+11*V3
                v2 = -V2+5*V3+2*V4
                v3 = 2*V3+5*V4-V5
                fm = invsum*(s11*v1+s22*v2+s33*v3)
                
                @inbounds flux[n] = (fp + fm) * tmp2
            end
            for n = 1:Ncons
                @inbounds F[i-NG, j-NG, k-NG, n] = flux[1] * R[n, 1] + flux[2] * R[n, 2] + 
                                                   flux[3] * R[n, 3] + flux[4] * R[n, 4] + 
                                                   flux[5] * R[n, 5]
            end
        else
            for n = 1:Ncons
                @inbounds fp = Fp[i, j-1, k, n] + 0.5f0*minmod(Fp[i, j, k, n] - Fp[i, j-1, k, n], 
                                                            Fp[i, j-1, k, n] - Fp[i, j-2, k, n])
                @inbounds fm = Fm[i, j, k, n] - 0.5f0*minmod(Fm[i, j+1, k, n] - Fm[i, j, k, n], 
                                                            Fm[i, j, k, n] - Fm[i, j-1, k, n])

                @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
            end 
        end
    end
    return
end

function advect_zc(F, ϕ, S, Fp, Fm, Q, Ax, Ay, Az)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG+1 || i < 1+NG || j < 1+NG || k < 1+NG
        return
    end

    tmp1::Float32 = 1/12f0
    tmp2::Float32 = 1/6f0

    WENOϵ1::Float64 = 1e-20
    WENOϵ2::Float32 = 1f-16

    c1::Float32 = UP7[1]
    c2::Float32 = UP7[2]
    c3::Float32 = UP7[3]
    c4::Float32 = UP7[4]
    c5::Float32 = UP7[5]
    c6::Float32 = UP7[6]
    c7::Float32 = UP7[7]

    # Jameson sensor
    @inbounds ϕx = max(ϕ[i, j, k-3], 
                       ϕ[i, j, k-2], 
                       ϕ[i, j, k-1], 
                       ϕ[i, j, k  ], 
                       ϕ[i, j, k+1], 
                       ϕ[i, j, k+2])

    if ϕx < hybrid_ϕ1
        for n = 1:Ncons
            @inbounds V1 = Fp[i, j, k-4, n]
            @inbounds V2 = Fp[i, j, k-3, n]
            @inbounds V3 = Fp[i, j, k-2, n]
            @inbounds V4 = Fp[i, j, k-1, n]
            @inbounds V5 = Fp[i, j, k,   n]
            @inbounds V6 = Fp[i, j, k+1, n]
            @inbounds V7 = Fp[i, j, k+2, n]
            
            fp = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6 + c7*V7

            @inbounds V1 = Fm[i, j, k+3, n]
            @inbounds V2 = Fm[i, j, k+2, n]
            @inbounds V3 = Fm[i, j, k+1, n]
            @inbounds V4 = Fm[i, j, k,   n]
            @inbounds V5 = Fm[i, j, k-1, n]
            @inbounds V6 = Fm[i, j, k-2, n]
            @inbounds V7 = Fm[i, j, k-3, n]

            fm = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6 + c7*V7

            @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
        end
    else
        @inbounds s0::Float32 = 2/(S[i, j, k-1] + S[i, j, k]) 

        # average
        @inbounds u = 0.5f0 * (Q[i, j, k, 2] + Q[i, j, k-1, 2])
        @inbounds v = 0.5f0 * (Q[i, j, k, 3] + Q[i, j, k-1, 3])
        @inbounds w = 0.5f0 * (Q[i, j, k, 4] + Q[i, j, k-1, 4])
        @inbounds T = 0.5f0 * (Q[i, j, k, 6] + Q[i, j, k-1, 6])
    
        @inbounds A1 = (Ax[i, j, k] + Ax[i, j, k-1]) * 0.5f0
        @inbounds A2 = (Ay[i, j, k] + Ay[i, j, k-1]) * 0.5f0
        @inbounds A3 = (Az[i, j, k] + Az[i, j, k-1]) * 0.5f0
    
        nx = A1*s0
        ny = A2*s0
        nz = A3*s0
    
        if abs(nz) <= abs(ny)
            ss = 1/sqrt(nx*nx+ny*ny) 
            lx = -ny*ss
            ly = nx*ss
            lz = 0.f0 
        else 
            ss = 1/sqrt(nx*nx+nz*nz)
            lx = -nz*ss
            ly = 0.f0   
            lz = nx*ss
        end 
        mx = ny*lz-nz*ly 
        my = nz*lx-nx*lz
        mz = nx*ly-ny*lx
    
        qn = u*nx+v*ny+w*nz 
        ql = u*lx+v*ly+w*lz 
        qm = u*mx+v*my+w*mz 
        q2 = 0.5f0*(u^2+v^2+w^2)
        c = sqrt(γ*Rg*T)
        c2 = 1/(2*c)
        K = (γ-1)/c^2 
        K2 = K * 0.5f0
        H = 1/K+q2
    
        L = MMatrix{Ncons, Ncons, Float32, Ncons*Ncons}(undef)
        R = MMatrix{Ncons, Ncons, Float32, Ncons*Ncons}(undef)
        flux = MVector{Ncons, Float32}(undef)
    
        L[1,1]=K2*q2+qn*c2;     L[1,2]=-(K2*u+nx*c2);     L[1,3]=-(K2*v+ny*c2);     L[1,4]=-(K2*w+nz*c2);     L[1,5]=K2
        L[2,1]=1-K*q2;          L[2,2]=K*u;               L[2,3]=K*v;               L[2,4]=K*w;               L[2,5]=-K 
        L[3,1]=K2*q2-qn*c2;     L[3,2]=-(K2*u-nx*c2);     L[3,3]=-(K2*v-ny*c2);     L[3,4]=-(K2*w-nz*c2);     L[3,5]=K2
        L[4,1]=-ql;             L[4,2]=lx;                L[4,3]=ly;                L[4,4]=lz;                L[4,5]=0.f0
        L[5,1]=-qm;             L[5,2]=mx;                L[5,3]=my;                L[5,4]=mz;                L[5,5]=0.f0
    
        R[1,1]=1.f0;      R[1,2]=1.f0;       R[1,3]=1.f0;      R[1,4]=0.f0;    R[1,5]=0.f0
        R[2,1]=u-c*nx;    R[2,2]=u;          R[2,3]=u+c*nx;    R[2,4]=lx;      R[2,5]=mx
        R[3,1]=v-c*ny;    R[3,2]=v;          R[3,3]=v+c*ny;    R[3,4]=ly;      R[3,5]=my		
        R[4,1]=w-c*nz;    R[4,2]=w;          R[4,3]=w+c*nz;    R[4,4]=lz;      R[4,5]=mz		
        R[5,1]=H-qn*c;    R[5,2]=q2;         R[5,3]=H+qn*c;    R[5,4]=ql;      R[5,5]=qm

        if ϕx < hybrid_ϕ2
            Fpc = MVector{7, Float32}(undef)
            Fmc = MVector{7, Float32}(undef)

            for n = 1:Ncons
                Fpc .= 0.f0
                Fmc .= 0.f0
                for m = 1:Ncons
                    ll = L[n, m]
                    @inbounds Fpc[1] += Fp[i, j, k-4, m] * ll
                    @inbounds Fpc[2] += Fp[i, j, k-3, m] * ll
                    @inbounds Fpc[3] += Fp[i, j, k-2, m] * ll
                    @inbounds Fpc[4] += Fp[i, j, k-1, m] * ll
                    @inbounds Fpc[5] += Fp[i, j, k,   m] * ll
                    @inbounds Fpc[6] += Fp[i, j, k+1, m] * ll
                    @inbounds Fpc[7] += Fp[i, j, k+2, m] * ll
        
                    @inbounds Fmc[1] += Fm[i, j, k-3, m] * ll
                    @inbounds Fmc[2] += Fm[i, j, k-2, m] * ll
                    @inbounds Fmc[3] += Fm[i, j, k-1, m] * ll
                    @inbounds Fmc[4] += Fm[i, j, k,   m] * ll
                    @inbounds Fmc[5] += Fm[i, j, k+1, m] * ll
                    @inbounds Fmc[6] += Fm[i, j, k+2, m] * ll
                    @inbounds Fmc[7] += Fm[i, j, k+3, m] * ll
                end

                @inbounds V1 = Fpc[1]
                @inbounds V2 = Fpc[2]
                @inbounds V3 = Fpc[3]
                @inbounds V4 = Fpc[4]
                @inbounds V5 = Fpc[5]
                @inbounds V6 = Fpc[6]
                @inbounds V7 = Fpc[7]
        
                # polinomia
                q1 = -3V1+13V2-23V3+25V4
                q2 = V2-5V3+13V4+3V5
                q3 = -V3+7V4+7V5-V6
                q4 = 3V4+13V5-5V6+V7
        
                # smoothness index
                Is1 = V1*( 547V1 - 3882V2 + 4642V3 - 1854V4) +
                    V2*(         7043V2 -17246V3 + 7042V4) +
                    V3*(                 11003V3 - 9402V4) +
                    V4*(                           2107V4)
                Is2 = V2*( 267V2 - 1642V3 + 1602V4 -  494V5) +
                    V3*(         2843V3 - 5966V4 + 1922V5) +
                    V4*(                  3443V4 - 2522V5) +
                    V5*(                            547V5)
                Is3 = V3*( 547V3 - 2522V4 + 1922V5 -  494V6) +
                    V4*(         3443V4 - 5966V5 + 1602V6) +
                    V5*(                  2843V5 - 1642V6) +
                    V6*(                            267V6)
                Is4 = V4*(2107V4 - 9402V5 + 7042V6 - 1854V7) +
                    V5*(        11003V5 -17246V6 + 4642V7) +
                    V6*(                  7043V6 - 3882V7) +
                    V7*(                            547V7)
        
                # alpha
                α1 = 1/(WENOϵ1+Is1*s0)^2
                α2 = 12/(WENOϵ1+Is2*s0)^2
                α3 = 18/(WENOϵ1+Is3*s0)^2
                α4 = 4/(WENOϵ1+Is4*s0)^2
        
                invsum = 1/(α1+α2+α3+α4)
        
                fp = invsum*(α1*q1+α2*q2+α3*q3+α4*q4)
        
                @inbounds V1 = Fmc[1]
                @inbounds V2 = Fmc[2]
                @inbounds V3 = Fmc[3]
                @inbounds V4 = Fmc[4]
                @inbounds V5 = Fmc[5]
                @inbounds V6 = Fmc[6]
                @inbounds V7 = Fmc[7]
        
                # polinomia
                q1 = -3V1+13V2-23V3+25V4
                q2 = V2-5V3+13V4+3V5
                q3 = -V3+7V4+7V5-V6
                q4 = 3V4+13V5-5V6+V7
        
                # smoothness index
                Is1 = V1*( 547V1 - 3882V2 + 4642V3 - 1854V4) +
                    V2*(         7043V2 -17246V3 + 7042V4) +
                    V3*(                 11003V3 - 9402V4) +
                    V4*(                           2107V4)
                Is2 = V2*( 267V2 - 1642V3 + 1602V4 -  494V5) +
                    V3*(         2843V3 - 5966V4 + 1922V5) +
                    V4*(                  3443V4 - 2522V5) +
                    V5*(                            547V5)
                Is3 = V3*( 547V3 - 2522V4 + 1922V5 -  494V6) +
                    V4*(         3443V4 - 5966V5 + 1602V6) +
                    V5*(                  2843V5 - 1642V6) +
                    V6*(                            267V6)
                Is4 = V4*(2107V4 - 9402V5 + 7042V6 - 1854V7) +
                    V5*(        11003V5 -17246V6 + 4642V7) +
                    V6*(                  7043V6 - 3882V7) +
                    V7*(                            547V7)
        
                # alpha
                α1 = 1/(WENOϵ1+Is1*s0)^2
                α2 = 12/(WENOϵ1+Is2*s0)^2
                α3 = 18/(WENOϵ1+Is3*s0)^2
                α4 = 4/(WENOϵ1+Is4*s0)^2
        
                invsum = 1/(α1+α2+α3+α4)
        
                fm = invsum*(α1*q1+α2*q2+α3*q3+α4*q4)
        
                @inbounds flux[n] = (fp + fm) * tmp1
            end

            for n = 1:Ncons
                @inbounds F[i-NG, j-NG, k-NG, n] = flux[1] * R[n, 1] + flux[2] * R[n, 2] + 
                                                   flux[3] * R[n, 3] + flux[4] * R[n, 4] + 
                                                   flux[5] * R[n, 5]
            end
        elseif ϕx < hybrid_ϕ3
            Fpc = MVector{5, Float32}(undef)
            Fmc = MVector{5, Float32}(undef)

            for n = 1:Ncons
                Fpc .= 0.f0
                Fmc .= 0.f0
                for m = 1:Ncons
                    ll = L[n, m]
                    @inbounds Fpc[1] += Fp[i, j, k-3, m] * ll
                    @inbounds Fpc[2] += Fp[i, j, k-2, m] * ll
                    @inbounds Fpc[3] += Fp[i, j, k-1, m] * ll
                    @inbounds Fpc[4] += Fp[i, j, k,   m] * ll
                    @inbounds Fpc[5] += Fp[i, j, k+1, m] * ll
        
                    @inbounds Fmc[1] += Fm[i, j, k-2, m] * ll
                    @inbounds Fmc[2] += Fm[i, j, k-1, m] * ll
                    @inbounds Fmc[3] += Fm[i, j, k,   m] * ll
                    @inbounds Fmc[4] += Fm[i, j, k+1, m] * ll
                    @inbounds Fmc[5] += Fm[i, j, k+2, m] * ll
                end

                @inbounds V1 = Fpc[1]
                @inbounds V2 = Fpc[2]
                @inbounds V3 = Fpc[3]
                @inbounds V4 = Fpc[4]
                @inbounds V5 = Fpc[5]
                # FP
                s11 = 13*(V1-2*V2+V3)^2 + 3*(V1-4*V2+3*V3)^2
                s22 = 13*(V2-2*V3+V4)^2 + 3*(V2-V4)^2
                s33 = 13*(V3-2*V4+V5)^2 + 3*(3*V3-4*V4+V5)^2

                s11 = 1/(WENOϵ2+s11*s0)^2
                s22 = 6/(WENOϵ2+s22*s0)^2
                s33 = 3/(WENOϵ2+s33*s0)^2

                invsum = 1/(s11+s22+s33)

                v1 = 2*V1-7*V2+11*V3
                v2 = -V2+5*V3+2*V4
                v3 = 2*V3+5*V4-V5
                fp = invsum*(s11*v1+s22*v2+s33*v3)

                @inbounds V1 = Fmc[1]
                @inbounds V2 = Fmc[2]
                @inbounds V3 = Fmc[3]
                @inbounds V4 = Fmc[4]
                @inbounds V5 = Fmc[5]
                # FM
                s11 = 13*(V1-2*V2+V3)^2 + 3*(V1-4*V2+3*V3)^2
                s22 = 13*(V2-2*V3+V4)^2 + 3*(V2-V4)^2
                s33 = 13*(V3-2*V4+V5)^2 + 3*(3*V3-4*V4+V5)^2

                s11 = 1/(WENOϵ2+s11*s0)^2
                s22 = 6/(WENOϵ2+s22*s0)^2
                s33 = 3/(WENOϵ2+s33*s0)^2

                invsum = 1/(s11+s22+s33)

                v1 = 2*V1-7*V2+11*V3
                v2 = -V2+5*V3+2*V4
                v3 = 2*V3+5*V4-V5
                fm = invsum*(s11*v1+s22*v2+s33*v3)
                
                @inbounds flux[n] = (fp + fm) * tmp2
            end
            for n = 1:Ncons
                @inbounds F[i-NG, j-NG, k-NG, n] = flux[1] * R[n, 1] + flux[2] * R[n, 2] + 
                                                   flux[3] * R[n, 3] + flux[4] * R[n, 4] + 
                                                   flux[5] * R[n, 5]
            end
        else
            for n = 1:Ncons
                @inbounds fp = Fp[i, j, k-1, n] + 0.5f0*minmod(Fp[i, j, k, n] - Fp[i, j, k-1, n], 
                                                            Fp[i, j, k-1, n] - Fp[i, j, k-2, n])
                @inbounds fm = Fm[i, j, k, n] - 0.5f0*minmod(Fm[i, j, k+1, n] - Fm[i, j, k, n], 
                                                            Fm[i, j, k, n] - Fm[i, j, k-1, n])

                @inbounds F[i-NG, j-NG, k-NG, n] = fp + fm
            end 
        end
    end
    return
end