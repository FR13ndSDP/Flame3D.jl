# using CUDA, Adapt, PyCall
# struct thermoProperty{IT, RT, VT, MT, TT}
#     Nspecs::IT
#     Ru::RT
#     min_temp::RT
#     max_temp::RT
#     mw::VT
#     coeffs_sep::VT
#     coeffs_lo::MT
#     coeffs_hi::MT
#     visc_poly::MT
#     conduct_poly::MT
#     binarydiff_poly::TT
# end
# Adapt.@adapt_structure thermoProperty

# const Nspecs = 5
using CUDA:i32
using StaticArrays

# get mixture pressure from T and ρi
@inline function Pmixture(T::Float64, ρi, thermo)
    YOW::Float64 = 0
    for n = 1:Nspecs
        @inbounds YOW += ρi[n]/thermo.mw[n]
    end
    return thermo.Ru * T * YOW
end

# get mixture density
@inline function ρmixture(P::Float64, T::Float64, Yi, thermo)
    YOW::Float64 = 0
    for n = 1:Nspecs
        @inbounds YOW += Yi[n] / thermo.mw[n]
    end
    return P/(thermo.Ru * T * YOW)
end

# mass fraction to mole fraction
@inline function Y2X(Yi, Xi, thermo)
    YOW::Float64 = 0

    for n = 1:Nspecs
        @inbounds YOW += Yi[n] / thermo.mw[n]
    end

    YOWINV::Float64 = 1/YOW

    for n = 1:Nspecs
        @inbounds Xi[n] = Yi[n] / thermo.mw[n] * YOWINV
    end
    
    return    
end

# mass fraction to mole fraction
@inline function ρi2X(ρi, Xi, thermo)
    ∑X::Float64 = 0
    for n = 1:Nspecs
        @inbounds Xi[n] = ρi[n] / thermo.mw[n]
        @inbounds ∑X += Xi[n]
    end
    
    ∑Xinv::Float64 = 1/∑X
    for n = 1:Nspecs
        @inbounds Xi[n] = Xi[n] * ∑Xinv
    end
    return    
end

# mole fraction to mass fraction
@inline function X2Y(Xi, Yi, thermo)
    XW::Float64 = 0
    for n = 1:Nspecs
        @inbounds XW += Xi[n] * thermo.mw[n]
    end
    for n = 1:Nspecs
        @inbounds Yi[n] = Xi[n] * thermo.mw[n] / XW
    end
    return    
end

# get cp for species using NASA-7 polynomial
@inline function cp_specs(cp, T::Float64, T2::Float64, T3::Float64, T4::Float64, thermo)
    for n = 1:Nspecs
        if T < thermo.coeffs_sep[n]
            @inbounds cp[n] = thermo.coeffs_lo[n, 1] +
                              thermo.coeffs_lo[n, 2] * T + 
                              thermo.coeffs_lo[n, 3] * T2 +
                              thermo.coeffs_lo[n, 4] * T3 + 
                              thermo.coeffs_lo[n, 5] * T4
        else
            @inbounds cp[n] = thermo.coeffs_hi[n, 1] + 
                              thermo.coeffs_hi[n, 2] * T + 
                              thermo.coeffs_hi[n, 3] * T2 +
                              thermo.coeffs_hi[n, 4] * T3 + 
                              thermo.coeffs_hi[n, 5] * T4
        end
    end

    return
end

# get enthalpy for species using NASA-7 polynomial, J/kg
@inline function h_specs(hi, T::Float64, T2::Float64, T3::Float64, T4::Float64, T5::Float64, thermo)
    for n = 1:Nspecs
        if T < thermo.coeffs_sep[n]
            @inbounds hi[n] = thermo.coeffs_lo[n, 1] * T +
                              thermo.coeffs_lo[n, 2] * 0.5 * T2 + 
                              thermo.coeffs_lo[n, 3] * 0.3333333333333333 * T3 + 
                              thermo.coeffs_lo[n, 4] * 0.25 * T4 + 
                              thermo.coeffs_lo[n, 5] * 0.2 * T5
        else
            @inbounds hi[n] = thermo.coeffs_hi[n, 1] * T + 
                              thermo.coeffs_hi[n, 2] * 0.5 * T2 + 
                              thermo.coeffs_hi[n, 3] * 0.3333333333333333 * T3 + 
                              thermo.coeffs_hi[n, 4] * 0.25 * T4 + 
                              thermo.coeffs_hi[n, 5] * 0.2 * T5 + 
                              (thermo.coeffs_hi[n, 6] - thermo.coeffs_lo[n, 6])
        end

        @inbounds hi[n] *= thermo.Ru / thermo.mw[n]
    end

    return
end

# get internal energy for species using NASA-7 polynomial
@inline function ei_specs(ei, T::Float64, T2::Float64, T3::Float64, T4::Float64, T5::Float64, thermo)
    for n = 1:Nspecs
        if T < thermo.coeffs_sep[n]
            @inbounds ei[n] = (thermo.coeffs_lo[n, 1] -1) * T + 
                               thermo.coeffs_lo[n, 2] * 0.5 * T2 + 
                               thermo.coeffs_lo[n, 3] * 0.3333333333333333 * T3 + 
                               thermo.coeffs_lo[n, 4] * 0.25 * T4 +
                               thermo.coeffs_lo[n, 5] * 0.2 * T5
        else
            @inbounds ei[n] = (thermo.coeffs_hi[n, 1] -1) * T + 
                               thermo.coeffs_hi[n, 2] * 0.5 * T2 + 
                               thermo.coeffs_hi[n, 3] * 0.3333333333333333 * T3 + 
                               thermo.coeffs_hi[n, 4] * 0.25 * T4 + 
                               thermo.coeffs_hi[n, 5] * 0.2 * T5 + 
                               (thermo.coeffs_hi[n, 6] - thermo.coeffs_lo[n, 6])
        end
    end

    return
end

# get gibbs free energy, gi/T, gi = g/Ri
@inline function gibbs(gi, logT::Float64, T::Float64, T2::Float64, T3::Float64, T4::Float64, thermo)
    invT::Float64 = 1/T

    for n = 1:Nspecs
        if T < thermo.coeffs_sep[n]
            @inbounds gi[n] = thermo.coeffs_lo[n, 1] * (1 - logT) - 
                              thermo.coeffs_lo[n, 2] * 0.5 * T - 
                              thermo.coeffs_lo[n, 3] * 0.1666666666666667 * T2 - 
                              thermo.coeffs_lo[n, 4] * 0.0833333333333333 * T3 -
                              thermo.coeffs_lo[n, 5] * 0.05 * T4 +
                              thermo.coeffs_lo[n, 6] * invT - 
                              thermo.coeffs_lo[n, 7]
        else
            @inbounds gi[n] = thermo.coeffs_hi[n, 1] * (1 - logT) - 
                              thermo.coeffs_hi[n, 2] * 0.5 * T - 
                              thermo.coeffs_hi[n, 3] * 0.1666666666666667 * T2 - 
                              thermo.coeffs_hi[n, 4] * 0.0833333333333333 * T3 -
                              thermo.coeffs_hi[n, 5] * 0.05 * T4 +
                              thermo.coeffs_hi[n, 6] * invT - 
                              thermo.coeffs_hi[n, 7]
        end
    end

    return
end

# TODO: reaction rate for air.yaml, make it more general
# O, O2, N, NO, N2
@inline function vproductionRate(wdot, sc, T, thermo)
    gi_T = MVector{Nspecs, Float64}(undef)
    k_f_s = MVector{5, Float64}(undef)
    Kc_s = MVector{5, Float64}(undef)
    q_f = MVector{5, Float64}(undef)
    q_r = MVector{5, Float64}(undef)
    vf = MMatrix{5, Nspecs, Int64}(undef)
    vr = MMatrix{5, Nspecs, Int64}(undef)

    lgT = log(T)
    T2 = T * T
    T3 = T2 * T
    T4 = T2 * T2
    invT = 1.0 / T
  
    # Ea, cal/mol to K: y = x * 4.184 / 8.314 ≈ y = x * 0.5032475342795285
    tmp::Float64 = 0.5032475342795285
    k_f_s[1] = 3.0e22 * exp(-1.6 * lgT - 224951.50535373 * tmp * invT)
    k_f_s[2] = 1.0e22 * exp(-1.5 * lgT - 117960.43602294 * tmp * invT)
    k_f_s[3] = 5.0e15 * exp(-150033.91037285 * tmp * invT)
    k_f_s[4] = 5.7e12 * exp(0.42 * lgT - 85326.57011377 * tmp * invT)
    k_f_s[5] = 8.4e12 * exp(-38551.75975143 * tmp * invT)
  
    # compute the Gibbs free energy 
  
    gibbs(gi_T, lgT, T, T2, T3, T4, thermo)
  
    RsT::Float64 = thermo.Ru / 101325.0 * 1e6 * T
  
    Kc_s[1] = 1.0/RsT * exp(gi_T[5]- 2 * gi_T[3])
    Kc_s[2] = 1.0/RsT * exp(gi_T[2]- 2 * gi_T[1])
    Kc_s[3] = 1.0/RsT * exp(gi_T[4]- (gi_T[1] + gi_T[3]))
    Kc_s[4] = exp((gi_T[1] + gi_T[5]) - (gi_T[3] + gi_T[4]))
    Kc_s[5] = exp((gi_T[1] + gi_T[4]) - (gi_T[2] + gi_T[3]))
  
    mixture::Float64 = 0.0
  
    for n = 1:Nspecs
        @inbounds mixture += sc[n]
        @inbounds wdot[n] = 0.0
    end
  
    for m = 1:5
        for n = 1:Nspecs
            @inbounds vf[m, n] = 0
            @inbounds vr[m, n] = 0
        end
    end

    # reaction 1: N2 + M <=> 2 N + M
    phi_f = sc[5]
    alpha = mixture - 0.76667 * (sc[5] + sc[4] + sc[2])
    k_f = alpha * k_f_s[1]
    q_f[1] = phi_f * k_f
    phi_r = sc[3] * sc[3]
    Kc = Kc_s[1]
    k_r = k_f / Kc
    q_r[1] = phi_r * k_r
    vf[1, 5] = 1
    vr[1, 3] = 2
  
    # reaction 2: O2 + M <=> 2 O + M
    phi_f = sc[2]
    alpha = mixture - 0.8 * (sc[5] + sc[4] + sc[2])
    k_f = alpha * k_f_s[2]
    q_f[2] = phi_f * k_f
    phi_r = sc[1] * sc[1]
    Kc = Kc_s[2]
    k_r = k_f / Kc
    q_r[2] = phi_r * k_r
    vf[2, 2] = 1
    vr[2, 1] = 2
  
    # reaction 3: NO + M <=> N + O + M
    phi_f = sc[4]
    alpha = mixture + 21 * (sc[4] + sc[3] + sc[1])
    k_f = alpha * k_f_s[3]
    q_f[3] = phi_f * k_f
    phi_r = sc[1] * sc[3]
    Kc = Kc_s[3]
    k_r = k_f / Kc;
    q_r[3] = phi_r * k_r
    vf[3, 4] = 1
    vr[3, 1] = 1
    vr[3, 3] = 1
  
    # reaction 4: N2 + O <=> NO + N
    phi_f = sc[1] * sc[5]
    k_f = k_f_s[4]
    q_f[4] = phi_f * k_f
    phi_r = sc[3] * sc[4]
    Kc = Kc_s[4]
    k_r = k_f / Kc
    q_r[4] = phi_r * k_r
    vf[4, 1] = 1
    vf[4, 5] = 1
    vr[4, 3] = 1
    vr[4, 4] = 1
  
    # reaction 5: NO + O <=> O2 + N
    phi_f = sc[1] * sc[4]
    k_f = k_f_s[5]
    q_f[5] = phi_f * k_f
    phi_r = sc[2] * sc[3]
    Kc = Kc_s[5]
    k_r = k_f / Kc
    q_r[5] = phi_r * k_r
    vf[5, 1] = 1
    vf[5, 4] = 1
    vr[5, 2] = 1
    vr[5, 3] = 1

    for m = 1:5
        @inbounds wf1 = q_f[m]
        @inbounds wr1 = q_r[m]
    
        for n = 1:Nspecs
            @inbounds wdot[n] += (wf1 - wr1) * (vr[m, n] - vf[m, n])
        end
    end
    return
end

@inline function vproductionRate_Jac(wdot, sc, Arate, T, thermo)
    gi_T = MVector{Nspecs, Float64}(undef)
    k_f_s = MVector{5, Float64}(undef)
    Kc_s = MVector{5, Float64}(undef)
    q_f = MVector{5, Float64}(undef)
    q_r = MVector{5, Float64}(undef)
    vf = MMatrix{5, Nspecs, Int64}(undef)
    vr = MMatrix{5, Nspecs, Int64}(undef)

    lgT = log(T)
    T2 = T * T
    T3 = T2 * T
    T4 = T2 * T2
    invT = 1.0 / T
  
    # Ea, cal/mol to K: y = x * 4.184 / 8.314 ≈ y = x * 0.5032475342795285
    tmp::Float64 = 0.5032475342795285
    k_f_s[1] = 3.0e22 * exp(-1.6 * lgT - 224951.50535373 * tmp * invT)
    k_f_s[2] = 1.0e22 * exp(-1.5 * lgT - 117960.43602294 * tmp * invT)
    k_f_s[3] = 5.0e15 * exp(-150033.91037285 * tmp * invT)
    k_f_s[4] = 5.7e12 * exp(0.42 * lgT - 85326.57011377 * tmp * invT)
    k_f_s[5] = 8.4e12 * exp(-38551.75975143 * tmp * invT)
  
    # compute the Gibbs free energy 
  
    gibbs(gi_T, lgT, T, T2, T3, T4, thermo)
  
    RsT::Float64 = thermo.Ru / 101325.0 * 1e6 * T
  
    Kc_s[1] = 1.0/RsT * exp(gi_T[5]- 2 * gi_T[3])
    Kc_s[2] = 1.0/RsT * exp(gi_T[2]- 2 * gi_T[1])
    Kc_s[3] = 1.0/RsT * exp(gi_T[4]- (gi_T[1] + gi_T[3]))
    Kc_s[4] = exp((gi_T[1] + gi_T[5]) - (gi_T[3] + gi_T[4]))
    Kc_s[5] = exp((gi_T[1] + gi_T[4]) - (gi_T[2] + gi_T[3]))
  
    mixture::Float64 = 0.0
  
    for n = 1:Nspecs
        @inbounds mixture += sc[n]
        @inbounds wdot[n] = 0.0
        for l = 1:Nspecs
            @inbounds Arate[n, l] = 0.0
        end
    end
  
    for m = 1:5
        for n = 1:Nspecs
            @inbounds vf[m, n] = 0
            @inbounds vr[m, n] = 0
        end
    end

    # reaction 1: N2 + M <=> 2 N + M
    phi_f = sc[5]
    alpha = mixture - 0.76667 * (sc[5] + sc[4] + sc[2])
    k_f = alpha * k_f_s[1]
    q_f[1] = phi_f * k_f
    phi_r = sc[3] * sc[3]
    Kc = Kc_s[1]
    k_r = k_f / Kc
    q_r[1] = phi_r * k_r
    vf[1, 5] = 1
    vr[1, 3] = 2
  
    # reaction 2: O2 + M <=> 2 O + M
    phi_f = sc[2]
    alpha = mixture - 0.8 * (sc[5] + sc[4] + sc[2])
    k_f = alpha * k_f_s[2]
    q_f[2] = phi_f * k_f
    phi_r = sc[1] * sc[1]
    Kc = Kc_s[2]
    k_r = k_f / Kc
    q_r[2] = phi_r * k_r
    vf[2, 2] = 1
    vr[2, 1] = 2
  
    # reaction 3: NO + M <=> N + O + M
    phi_f = sc[4]
    alpha = mixture + 21 * (sc[4] + sc[3] + sc[1])
    k_f = alpha * k_f_s[3]
    q_f[3] = phi_f * k_f
    phi_r = sc[1] * sc[3]
    Kc = Kc_s[3]
    k_r = k_f / Kc;
    q_r[3] = phi_r * k_r
    vf[3, 4] = 1
    vr[3, 1] = 1
    vr[3, 3] = 1
  
    # reaction 4: N2 + O <=> NO + N
    phi_f = sc[1] * sc[5]
    k_f = k_f_s[4]
    q_f[4] = phi_f * k_f
    phi_r = sc[3] * sc[4]
    Kc = Kc_s[4]
    k_r = k_f / Kc
    q_r[4] = phi_r * k_r
    vf[4, 1] = 1
    vf[4, 5] = 1
    vr[4, 3] = 1
    vr[4, 4] = 1
  
    # reaction 5: NO + O <=> O2 + N
    phi_f = sc[1] * sc[4]
    k_f = k_f_s[5]
    q_f[5] = phi_f * k_f
    phi_r = sc[2] * sc[3]
    Kc = Kc_s[5]
    k_r = k_f / Kc
    q_r[5] = phi_r * k_r
    vf[5, 1] = 1
    vf[5, 4] = 1
    vr[5, 2] = 1
    vr[5, 3] = 1

    for m = 1:5
        @inbounds wf1 = q_f[m]
        @inbounds wr1 = q_r[m]
    
        for n = 1:Nspecs
            @inbounds wdot[n] += (wf1 - wr1) * (vr[m, n] - vf[m, n])
        end

        for n = 1:Nspecs
            @inbounds Awf = vf[m, n] * wf1 / (sc[n] + eps(Float64))
            @inbounds Awr = vr[m, n] * wr1 / (sc[n] + eps(Float64))
            for l = 1:Nspecs
                @inbounds Arate[l, n] += (Awf - Awr) * (vr[m, l] - vf[m, l])
            end
        end
    end
    return
end

# Solve Ax = b with Gauss Elimination
@inline function solve(x, A, b)
    U = MMatrix{Nspecs, Nspecs+1, Float64}(undef)

    # Copy A to U and augment with vector b.
    for ii = 1:Nspecs
        @inbounds U[ii, Nspecs+1] = b[ii]
        for jj = 1:Nspecs
            @inbounds U[ii, jj] = A[ii, jj]
        end
    end
  
    # Factorisation stage
    for kk = 1:Nspecs
        # Find the best pivot
        p = kk
        maxPivot::Float64 = 0.0
        for ii = kk:Nspecs
            if (@inbounds abs(U[ii, kk]) > maxPivot)
                @inbounds maxPivot = abs(U[ii, kk])
                p = ii
            end
        end
        # Swap rows kk and p
        if (p != kk) 
            for ii = kk:Nspecs+1
                @inbounds tmp = U[p, ii]
                @inbounds U[p, ii] = U[kk, ii]
                @inbounds U[kk, ii] = tmp
            end
        end
      
  
        # Elimination of variables
        for ii = kk+1:Nspecs
            @inbounds m = U[ii, kk] / U[kk, kk]
            for jj = kk:Nspecs+1
                @inbounds U[ii, jj] -= m * U[kk, jj]
            end
        end
    end
  
    # Back substitution
    for kk = Nspecs:-1:1
        @inbounds sum = U[kk, Nspecs+1]
        for jj = kk+1:Nspecs
            @inbounds sum -= U[kk, jj] * x[jj]
        end
        @inbounds x[kk] = sum / U[kk, kk]
    end
    return
end

# J/(m^3 K)
@inline function CV(T::Float64, rhoi, thermo)
    T2 = T*T
    T3 = T2*T
    T4 = T2*T2

    cp = MVector{Nspecs, Float64}(undef)
    cp_specs(cp, T, T2, T3, T4, thermo)

    result::Float64 = 0
    for n = 1:Nspecs
        @inbounds result += (cp[n] - 1)*rhoi[n]/thermo.mw[n]
    end
    return result*thermo.Ru
end

# J/(m^3 K)
@inline function CP(T::Float64, rhoi, thermo)
    T2 = T*T
    T3 = T2*T
    T4 = T2*T2

    cp = MVector{Nspecs, Float64}(undef)
    cp_specs(cp, T, T2, T3, T4, thermo)

    result::Float64 = 0
    for n = 1:Nspecs
        @inbounds result += cp[n]*rhoi[n]/thermo.mw[n]
    end
    return result*thermo.Ru
end

# get mean internal energy in volume unit
# J/m^3
@inline function InternalEnergy(T::Float64, rhoi, thermo)
    T2 = T*T
    T3 = T2*T
    T4 = T2*T2
    T5 = T3*T2

    ei = MVector{Nspecs, Float64}(undef)
    ei_specs(ei, T, T2, T3, T4, T5, thermo)

    result::Float64 = 0
    for n = 1:Nspecs
        @inbounds result += rhoi[n] * ei[n]/thermo.mw[n]
    end
    return result * thermo.Ru
end

# get temperature from ρi and internal energy
@inline function GetT(ein::Float64, ρi, thermo)
    maxiter::Int32 = 30
    tol::Float64 = 1e-3
    tmin::Float64 = thermo.min_temp
    tmax::Float64 = thermo.max_temp

    emin = InternalEnergy(tmin, ρi, thermo)
    emax = InternalEnergy(tmax, ρi, thermo)
    if ein < emin
      # Linear Extrapolation below tmin
      cv = CV(tmin, ρi, thermo)
      T = tmin - (emin - ein) / cv
      return T
    end

    if ein > emax
      # Linear Extrapolation above tmax
      cv = CV(tmax, ρi, thermo)
      T = tmax - (emax - ein) / cv
      return T
    end
  
    As::Float64=0
    for n = 1:Nspecs
        @inbounds As += (thermo.coeffs_lo[n, 1]-1) *thermo.Ru/thermo.mw[n]*ρi[n]
    end
  
    # initial value
    t1::Float64 = ein/As

    if t1 < tmin || t1 > tmax
        t1 = tmin + (tmax - tmin) / (emax - emin) * (ein - emin)
    end
  
    for _ = 1:maxiter
        e1 = InternalEnergy(t1, ρi, thermo)
        cv = CV(t1, ρi, thermo)

        dt = (ein - e1) / cv
        if dt > 100.0
            dt = 100.0
        elseif dt < -100.0
            dt = -100.0
        elseif (CUDA.abs(dt) < tol)
            break
        elseif (t1+dt == t1)
            break
        end
        t1 += dt
    end
    return t1
end

@inline function dot5(lgT, lgT2, lgT3, lgT4, poly)
    return poly[1] + lgT*poly[2] + lgT2*poly[3] + lgT3*poly[4] + lgT4*poly[5]
end

# compute mixture viscosity and heat conduct coeff
@inline function mixtureProperties(T, P, X, μi, D, Diff, thermo)
    sqT::Float64 = sqrt(T)
    sqsqT::Float64 = sqrt(sqT)
    lgT = log(T)
    lgT2 = lgT * lgT
    lgT3 = lgT * lgT2
    lgT4 = lgT2 * lgT2

    # λ
    for n = 1:Nspecs
        @inbounds μi[n] = sqT * dot5(lgT, lgT2, lgT3, lgT4, @inbounds @view thermo.conduct_poly[n, :])
    end

    sum1::Float64 = 0
    sum2::Float64 = 0
    for k = 1:Nspecs
        @inbounds sum1 += X[k] * μi[k]
        @inbounds sum2 += X[k] / μi[k]
    end
    λ::Float64 = 0.5*(sum1 + 1/sum2)

    # μ
    for n = 1:Nspecs
        # the polynomial fit is done for sqrt(visc/sqrt(T))
        sqmui = sqsqT * dot5(lgT, lgT2, lgT3, lgT4, @inbounds @view thermo.visc_poly[n, :])
        @inbounds μi[n] = (sqmui * sqmui)
    end

    # Wilke fit, see Eq. (9-5.14) of Poling et al. (2001)
    for n = 1:Nspecs
        for l = 1:n
            @inbounds wratioln = thermo.mw[l]/thermo.mw[n]
            @inbounds vrationl = μi[n]/μi[l]

            @inbounds factor1 = 1 + CUDA.sqrt(vrationl * CUDA.sqrt(wratioln))
            @inbounds tmp = factor1*factor1 / CUDA.sqrt(8+8*thermo.mw[n]/thermo.mw[l])
            @inbounds D[(n-1)*Nspecs+l] = tmp
            @inbounds D[(l-1)*Nspecs+n] = tmp / (vrationl * wratioln)
        end
        @inbounds D[(n-1)*Nspecs+n] = 1.0
    end

    μ::Float64 = 0
    for n = 1:Nspecs
        tmp = 0
        for l = 1:Nspecs
            @inbounds tmp += X[l] * D[(n-1)*Nspecs+l]
        end
        @inbounds μ += X[n]*μi[n]/tmp
    end

    # D
    #= 
    get the mixture-averaged diffusion coefficients [m^2/s].
    =#
    for n = 1:Nspecs
        for nn = n:Nspecs
            tmp = T * sqT *dot5(lgT, lgT2, lgT3, lgT4, @inbounds @view thermo.binarydiff_poly[n, nn, :])
            @inbounds D[(nn-1)*Nspecs+n] = tmp
            @inbounds D[(n-1)*Nspecs+nn] = tmp
        end
    end
 
    for n = 1:Nspecs
        sum1 = 0
        for nn = 1:Nspecs
            if nn == n
                continue
            end
            @inbounds sum1 += X[nn] / D[(n-1)*Nspecs+nn]
        end
        sum1 *= P
        @inbounds Diff[n] = (1-X[n])/(sum1+eps(Float64))
    end
    return λ, μ
end

function mixture(Q, Yi, λ, μ, D, thermo)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    X1 = MVector{Nspecs, Float64}(undef)
    μ1 = MVector{Nspecs, Float64}(undef)
    D1 = MVector{Nspecs*Nspecs, Float64}(undef)

    @inbounds T = Q[i, j, k, 6]
    @inbounds P = Q[i, j, k, 5]

    Y1 = @inbounds @view Yi[i, j, k, :]
    diff = @inbounds @view D[i, j, k, :]
    Y2X(Y1, X1, thermo)

    lambda, mu = mixtureProperties(T, P, X1, μ1, D1, diff, thermo)
    @inbounds λ[i, j, k] = lambda
    @inbounds μ[i, j, k] = mu
    return
end

function initThermo(mech)
    ct = pyimport("cantera")
    gas = ct.Solution(mech)

    Ru = ct.gas_constant * 1e-3
    mw = gas.molecular_weights * 1e-3
    min_temp = gas.min_temp
    max_temp = gas.max_temp
    coeffs_sep = zeros(Float64, Nspecs)
    coeffs_hi = zeros(Float64, Nspecs, 7)
    coeffs_lo = zeros(Float64, Nspecs, 7)
    viscosity_poly = zeros(Float64, Nspecs, 5)
    conductivity_poly = zeros(Float64, Nspecs, 5)
    binarydiffusion_poly = zeros(Float64, Nspecs, Nspecs, 5)

    for i = 1:Nspecs
        coeffs_sep[i] = gas.species(i-1).thermo.coeffs[1]
        coeffs_hi[i, :] = gas.species(i-1).thermo.coeffs[2:8]
        coeffs_lo[i, :] = gas.species(i-1).thermo.coeffs[9:end]
        viscosity_poly[i, :] = gas.get_viscosity_polynomial(i-1)
        conductivity_poly[i, :] = gas.get_thermal_conductivity_polynomial(i-1)
        for j = 1:Nspecs
            binarydiffusion_poly[i, j, :] = gas.get_binary_diff_coeffs_polynomial(i-1, j-1)
        end
    end

    thermo = thermoProperty(Nspecs, Ru, min_temp, max_temp, CuArray(mw),
                            CuArray(coeffs_sep), CuArray(coeffs_lo), CuArray(coeffs_hi), 
                            CuArray(viscosity_poly), CuArray(conductivity_poly), CuArray(binarydiffusion_poly))
    return thermo
end


# mech = "./NN/Air/air.yaml"
# ct = pyimport("cantera")
# gas = ct.Solution(mech)
# T::Float64 = 1000
# P::Float64 = 3596
# gas.TPY = T, P, "N2:0.77 O2:0.23"

# ρi = gas.Y * gas.density
# ρi_d = CuArray(ρi)
# tmp = CUDA.zeros(Float64, Nspecs)
# thermo = initThermo(mech)
# ei = InternalEnergy(T, ρi, thermo)
# @show ei
# # # @cuda threads=16 GetT(ei, ρi_d, thermo)