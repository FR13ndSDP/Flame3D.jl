# using CUDA, Adapt, PyCall, StaticArrays
# struct thermoProperty{RT, VT, MT, TT}
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

# CUDA.allowscalar(true)
# const Nspecs = 20

# get mixture pressure from T and ρi
@inline function Pmixture(T::Float32, ρi, thermo)
    YOW::Float32 = 0
    for n = 1:Nspecs
        @inbounds YOW += ρi[n]/thermo.mw[n]
    end
    return thermo.Ru * T * YOW
end

# get mixture density
@inline function ρmixture(P::Float32, T::Float32, Yi, thermo)
    YOW::Float32 = 0
    for n = 1:Nspecs
        @inbounds YOW += Yi[n] / thermo.mw[n]
    end
    return P/(thermo.Ru * T * YOW)
end

# mass fraction to mole fraction
@inline function Y2X(Yi, Xi, thermo)
    YOW::Float32 = 0

    for n = 1:Nspecs
        @inbounds YOW += Yi[n] / thermo.mw[n]
    end

    YOWINV::Float32 = 1/YOW

    for n = 1:Nspecs
        @inbounds Xi[n] = Yi[n] / thermo.mw[n] * YOWINV
    end
    
    return    
end

# mass fraction to mole fraction
@inline function ρi2X(ρi, Xi, thermo)
    ∑X::Float32 = 0
    for n = 1:Nspecs
        @inbounds Xi[n] = ρi[n] / thermo.mw[n]
        @inbounds ∑X += Xi[n]
    end
    
    ∑Xinv::Float32 = 1/∑X
    for n = 1:Nspecs
        @inbounds Xi[n] = Xi[n] * ∑Xinv
    end
    return    
end

# mole fraction to mass fraction
@inline function X2Y(Xi, Yi, thermo)
    XW::Float32 = 0
    for n = 1:Nspecs
        @inbounds XW += Xi[n] * thermo.mw[n]
    end
    for n = 1:Nspecs
        @inbounds Yi[n] = Xi[n] * thermo.mw[n] / XW
    end
    return    
end

# get cp for species using NASA-7 polynomial
@inline function cp_specs(cp, T::Float32, thermo)
    T2::Float32 = T * T
    T3::Float32 = T2 * T
    T4::Float32 = T2 * T2

    for n = 1:Nspecs
        if T < thermo.coeffs_sep[n]
            @inbounds cp[n] = thermo.coeffs_lo[1, n] +
                              thermo.coeffs_lo[2, n] * T + 
                              thermo.coeffs_lo[3, n] * T2 +
                              thermo.coeffs_lo[4, n] * T3 + 
                              thermo.coeffs_lo[5, n] * T4
        else
            @inbounds cp[n] = thermo.coeffs_hi[1, n] + 
                              thermo.coeffs_hi[2, n] * T + 
                              thermo.coeffs_hi[3, n] * T2 +
                              thermo.coeffs_hi[4, n] * T3 + 
                              thermo.coeffs_hi[5, n] * T4
        end
    end

    return
end

# get enthalpy for species using NASA-7 polynomial, J/kg
@inline function h_specs(hi, T::Float32, thermo)
    T2::Float32 = 0.5f0 * T * T # 1/2T^2
    T3::Float32 = 2/3f0 * T2 * T # 1/3T^3
    T4::Float32 = T2 * T2 # 1/4T^4
    T5::Float32 = 0.8f0 * T4 * T # 1/5T^5

    for n = 1:Nspecs
        if T < thermo.coeffs_sep[n]
            @inbounds hi[n] = thermo.coeffs_lo[1, n] * T +
                              thermo.coeffs_lo[2, n] * T2 + 
                              thermo.coeffs_lo[3, n] * T3 + 
                              thermo.coeffs_lo[4, n] * T4 + 
                              thermo.coeffs_lo[5, n] * T5
        else
            @inbounds hi[n] = thermo.coeffs_hi[1, n] * T + 
                              thermo.coeffs_hi[2, n] * T2 + 
                              thermo.coeffs_hi[3, n] * T3 + 
                              thermo.coeffs_hi[4, n] * T4 + 
                              thermo.coeffs_hi[5, n] * T5 + 
                              (thermo.coeffs_hi[6, n] - thermo.coeffs_lo[6, n])
        end

        @inbounds hi[n] *= thermo.Ru / thermo.mw[n]
    end

    return
end

# get internal energy for species using NASA-7 polynomial
@inline function ei_specs(ei, T::Float32, thermo)
    T2::Float32 = 0.5f0 * T * T # 1/2T^2
    T3::Float32 = 2/3f0 * T2 * T # 1/3T^3
    T4::Float32 = T2 * T2 # 1/4T^4
    T5::Float32 = 0.8f0 * T4 * T # 1/5T^5

    for n = 1:Nspecs
        if T < thermo.coeffs_sep[n]
            @inbounds ei[n] = (thermo.coeffs_lo[1, n] -1) * T + 
                               thermo.coeffs_lo[2, n] * T2 + 
                               thermo.coeffs_lo[3, n] * T3 + 
                               thermo.coeffs_lo[4, n] * T4 +
                               thermo.coeffs_lo[5, n] * T5
        else
            @inbounds ei[n] = (thermo.coeffs_hi[1, n] -1) * T + 
                               thermo.coeffs_hi[2, n] * T2 + 
                               thermo.coeffs_hi[3, n] * T3 + 
                               thermo.coeffs_hi[4, n] * T4 + 
                               thermo.coeffs_hi[5, n] * T5 + 
                               (thermo.coeffs_hi[6, n] - thermo.coeffs_lo[6, n])
        end
    end

    return
end

# get gibbs free energy, gi/T, gi = g/Ri
@inline function gibbs(gi, T::Float32, lgT::Float32, invT::Float32, thermo)
    mlogT::Float32 = 1-lgT
    T1::Float32 = 0.5f0 * T # 1/2T
    T2::Float32 = 1/6f0 * T * T # 1/6T^2
    T3::Float32 = T1 * T2 # 1/12T^3
    T4::Float32 = 1.8f0 * T2 * T2 # 1/20T^4

    for n = 1:Nspecs
        if T < thermo.coeffs_sep[n]
            @inbounds gi[n] = thermo.coeffs_lo[1, n] * mlogT - 
                              thermo.coeffs_lo[2, n] * T1 - 
                              thermo.coeffs_lo[3, n] * T2 - 
                              thermo.coeffs_lo[4, n] * T3 -
                              thermo.coeffs_lo[5, n] * T4 +
                              thermo.coeffs_lo[6, n] * invT - 
                              thermo.coeffs_lo[7, n]
        else
            @inbounds gi[n] = thermo.coeffs_hi[1, n] * mlogT - 
                              thermo.coeffs_hi[2, n] * T1 - 
                              thermo.coeffs_hi[3, n] * T2 - 
                              thermo.coeffs_hi[4, n] * T3 -
                              thermo.coeffs_hi[5, n] * T4 +
                              thermo.coeffs_hi[6, n] * invT - 
                              thermo.coeffs_hi[7, n]
        end
    end

    return
end

# J/(m^3 K)
@inline function CV(T::Float32, rhoi, thermo)
    cp = MVector{Nspecs, Float32}(undef)
    cp_specs(cp, T, thermo)

    result::Float32 = 0
    for n = 1:Nspecs
        @inbounds result += (cp[n] - 1)*rhoi[n]/thermo.mw[n]
    end
    return result*thermo.Ru
end

# J/(m^3 K)
@inline function CP(T::Float32, rhoi, thermo)
    cp = MVector{Nspecs, Float32}(undef)
    cp_specs(cp, T, thermo)

    result::Float32 = 0
    for n = 1:Nspecs
        @inbounds result += cp[n]*rhoi[n]/thermo.mw[n]
    end
    return result*thermo.Ru
end

# get mean internal energy in volume unit
# J/m^3
@inline function InternalEnergy(T::Float32, rhoi, thermo)
    ei = MVector{Nspecs, Float32}(undef)
    ei_specs(ei, T, thermo)

    result::Float32 = 0
    for n = 1:Nspecs
        @inbounds result += rhoi[n] * ei[n]/thermo.mw[n]
    end
    return result * thermo.Ru
end

# get temperature from ρi and internal energy
@inline function GetT(ein::Float32, ρi, thermo)
    maxiter::Int32 = 30
    tol::Float32 = 1f-3
    tmin::Float32 = thermo.min_temp
    tmax::Float32 = thermo.max_temp

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
  
    As::Float32=0
    for n = 1:Nspecs
        @inbounds As += (thermo.coeffs_lo[1, n]-1) *thermo.Ru/thermo.mw[n]*ρi[n]
    end
  
    # initial value
    t1::Float32 = ein/As

    if t1 < tmin || t1 > tmax
        t1 = tmin + (tmax - tmin) / (emax - emin) * (ein - emin)
    end
  
    for _ = 1:maxiter
        e1 = InternalEnergy(t1, ρi, thermo)
        cv = CV(t1, ρi, thermo)

        dt = (ein - e1) / cv
        if dt > 100.f0
            dt = 100.f0
        elseif dt < -100.f0
            dt = -100.f0
        elseif (abs(dt) < tol)
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
@inline function mixtureProperties(T, P, X, Diff, thermo)
    μi = MVector{Nspecs, Float32}(undef)
    D = MVector{Nspecs*Nspecs, Float32}(undef)

    @fastmath sqT::Float32 = sqrt(T)
    @fastmath sqsqT::Float32 = sqrt(sqT)
    @fastmath lgT = log(T)
    lgT2 = lgT * lgT
    lgT3 = lgT * lgT2
    lgT4 = lgT2 * lgT2

    # λ
    for n = 1:Nspecs
        @inbounds μi[n] = sqT * dot5(lgT, lgT2, lgT3, lgT4, @inbounds @view thermo.conduct_poly[:, n])
    end

    sum1::Float32 = 0
    sum2::Float32 = 0
    for k = 1:Nspecs
        @inbounds sum1 += X[k] * μi[k]
        @inbounds sum2 += X[k] / μi[k]
    end
    λ::Float32 = 0.5f0*(sum1 + 1/sum2)

    # μ
    for n = 1:Nspecs
        # the polynomial fit is done for sqrt(visc/sqrt(T))
        sqmui = sqsqT * dot5(lgT, lgT2, lgT3, lgT4, @inbounds @view thermo.visc_poly[:, n])
        @inbounds μi[n] = (sqmui * sqmui)
    end

    # Wilke fit, see Eq. (9-5.14) of Poling et al. (2001)
    for n = 1:Nspecs
        tmp1::Float32 = 1/thermo.mw[n]
        tmp2::Float32 = μi[n]
        for l = 1:n
            @inbounds wratioln = thermo.mw[l]*tmp1
            @inbounds vrationl = tmp2/μi[l]

            @inbounds @fastmath factor1 = 1 + sqrt(vrationl * sqrt(wratioln))
            @inbounds @fastmath tmp = factor1*factor1 / sqrt(8+8/wratioln)
            @inbounds D[(n-1)*Nspecs+l] = tmp
            @inbounds D[(l-1)*Nspecs+n] = tmp / (vrationl * wratioln)
        end
        @inbounds D[(n-1)*Nspecs+n] = 1.f0
    end

    μ::Float32 = 0
    for n = 1:Nspecs
        tmp::Float32 = 0.f0
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
            tmp = T * sqT *dot5(lgT, lgT2, lgT3, lgT4, @inbounds @view thermo.binarydiff_poly[:, nn, n])
            @inbounds D[(nn-1)*Nspecs+n] = tmp
            @inbounds D[(n-1)*Nspecs+nn] = tmp
        end
    end
 
    for n = 1:Nspecs
        sum1 = 0.f0
        for nn = 1:Nspecs
            if nn == n
                continue
            end
            @inbounds sum1 += X[nn] / D[(n-1)*Nspecs+nn]
        end
        sum1 *= P
        @inbounds Diff[n] = (1-X[n])/(sum1+1f-30)
    end
    return λ, μ
end

function mixture(Q, ρi, Yi, λ, μ, D, thermo)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
        return
    end

    X1 = MVector{Nspecs, Float32}(undef)

    @inbounds T = Q[i, j, k, 6]
    @inbounds P = Q[i, j, k, 5]

    Y1 = @inbounds @view Yi[i, j, k, :]

    @inbounds ρinv::Float32 = 1/max(Q[i, j, k, 1], CUDA.eps(Float32))
    for n = 1:Nspecs
        @inbounds Y1[n] = max(ρi[i, j, k, n]*ρinv, 0.f0)
    end

    diff = @inbounds @view D[i, j, k, :]
    Y2X(Y1, X1, thermo)

    lambda, mu = mixtureProperties(T, P, X1, diff, thermo)
    @inbounds λ[i, j, k] = lambda
    @inbounds μ[i, j, k] = mu
    return
end

function initThermo(mech)
    ct = pyimport("cantera")
    gas = ct.Solution(mech)

    Ru::Float32 = ct.gas_constant * 1e-3
    min_temp::Float32 = gas.min_temp
    max_temp::Float32 = gas.max_temp
    mw = gas.molecular_weights * 1e-3
    coeffs_sep = zeros(Float64, Nspecs)
    coeffs_hi = zeros(Float64, 7, Nspecs)
    coeffs_lo = zeros(Float64, 7, Nspecs)
    viscosity_poly = zeros(Float64, 5, Nspecs)
    conductivity_poly = zeros(Float64, 5, Nspecs)
    binarydiffusion_poly = zeros(Float64, 5, Nspecs, Nspecs)

    for j = 1:Nspecs
        spec_i = gas.species(j-1)
        coeffs_sep[j] = spec_i.thermo.coeffs[1]
        coeffs_hi[:, j] = spec_i.thermo.coeffs[2:8]
        coeffs_lo[:, j] = spec_i.thermo.coeffs[9:end]
        viscosity_poly[:, j] = gas.get_viscosity_polynomial(j-1)
        conductivity_poly[:, j] = gas.get_thermal_conductivity_polynomial(j-1)
        for i = 1:Nspecs
            binarydiffusion_poly[:, i, j] = gas.get_binary_diff_coeffs_polynomial(i-1, j-1)
        end
    end

    thermo = thermoProperty(Ru, min_temp, max_temp, cu(mw),
                            cu(coeffs_sep), cu(coeffs_lo), cu(coeffs_hi), 
                            cu(viscosity_poly), cu(conductivity_poly), cu(binarydiffusion_poly))
    return thermo
end


# mech = "./drm19.yaml"
# ct = pyimport("cantera")
# gas = ct.Solution(mech)
# T::Float32 = 290.0
# P::Float32 = 101325.0
# gas.TPY = T, P, "CH4:1"

# ρi = gas.Y * gas.density
# ρi_d = cu(ρi)
# thermo = initThermo(mech)
# ei = InternalEnergy(T, ρi, thermo)
# @show ei