using StaticArrays

# using CUDA, Adapt, PyCall
# struct thermoProperty{IT, RT, VT, MT, TT}
#     Nspecs::IT
#     Ru::RT
#     mw::VT
#     coeffs_sep::VT
#     coeffs_lo::MT
#     coeffs_hi::MT
#     visc_poly::MT
#     conduct_poly::MT
#     binarydiff_poly::TT
# end

# Adapt.@adapt_structure thermoProperty

# const Nspecs = 8

# get mixture pressure from T and ρi
@inline function Pmixture(T, ρi, thermo)
    YOW::Float64 = 0
    for n = 1:Nspecs
        YOW += ρi[n]/thermo.mw[n]
    end
    return thermo.Ru * T * YOW
end

# get mixture density
@inline function ρmixture(P, T, Yi, thermo)
    YOW::Float64 = 0
    for n = 1:Nspecs
        YOW += Yi[n] / thermo.mw[n]
    end
    return P/(thermo.Ru * T * YOW)
end

# mass fraction to mole fraction
@inline function Y2X(Yi, Xi, thermo)
    YOW::Float64 = 0

    for n = 1:Nspecs
        YOW += Yi[n] / thermo.mw[n]
    end

    YOWINV = 1/YOW

    for n = 1:Nspecs
        Xi[n] = Yi[n] / thermo.mw[n] * YOWINV
    end
    
    return    
end

# mole fraction to mass fraction
@inline function X2Y(Xi, Yi, thermo)
    XW::Float64 = 0
    for n = 1:Nspecs
        XW += Xi[n] * thermo.mw[n]
    end
    for n = 1:Nspecs
        Yi[n] = Xi[n] * thermo.mw[n] / XW
    end
    return    
end

# get cp for species using NASA-7 polynomial
@inline function cp_specs(cp, tc, thermo)
    # temperature
    T = tc[1]

    for n = 1:Nspecs
        if T < thermo.coeffs_sep[n]
            cp[n] = thermo.coeffs_lo[n, 1] + thermo.coeffs_lo[n, 2] * tc[1] + thermo.coeffs_lo[n, 3] * tc[2] +
            thermo.coeffs_lo[n, 4] * tc[3] + thermo.coeffs_lo[n, 5] * tc[4]
        else
            cp[n] = thermo.coeffs_hi[n, 1] + thermo.coeffs_hi[n, 2] * tc[1] + thermo.coeffs_hi[n, 3] * tc[2] +
            thermo.coeffs_hi[n, 4] * tc[3] + thermo.coeffs_hi[n, 5] * tc[4]
        end
    end

    return
end

# get enthalpy for species using NASA-7 polynomial
@inline function h_specs(hi, T, thermo)
    # temperature
    tc = SVector{5, Float64}(T, T^2, T^3, T^4, T^5)

    for n = 1:Nspecs
        if T < thermo.coeffs_sep[n]
            hi[n] = thermo.coeffs_lo[n, 1] * tc[1] + thermo.coeffs_lo[n, 2]/2 * tc[2] + thermo.coeffs_lo[n, 3]/3 * tc[3] +
            thermo.coeffs_lo[n, 4]/4 * tc[4] + thermo.coeffs_lo[n, 5]/5 * tc[5]
        else
            hi[n] = thermo.coeffs_hi[n, 1] * tc[1] + thermo.coeffs_hi[n, 2]/2 * tc[2] + thermo.coeffs_hi[n, 3]/3 * tc[3] +
            thermo.coeffs_hi[n, 4]/4 * tc[4] + thermo.coeffs_hi[n, 5]/5 * tc[5] + (thermo.coeffs_hi[n, 6] - thermo.coeffs_lo[n, 6])
        end

        ei[n] *= thermo.Ru/thermo.mw[n]
    end

    return
end

# get internal energy for species using NASA-7 polynomial
@inline function ei_specs(ei, tc, thermo)
    # temperature
    T = tc[1]

    for n = 1:Nspecs
        if T < thermo.coeffs_sep[n]
            ei[n] = (thermo.coeffs_lo[n, 1] -1) * tc[1] + thermo.coeffs_lo[n, 2]/2 * tc[2] + thermo.coeffs_lo[n, 3]/3 * tc[3] +
            thermo.coeffs_lo[n, 4]/4 * tc[4] + thermo.coeffs_lo[n, 5]/5 * tc[5]
        else
            ei[n] = (thermo.coeffs_hi[n, 1] -1) * tc[1] + thermo.coeffs_hi[n, 2]/2 * tc[2] + thermo.coeffs_hi[n, 3]/3 * tc[3] +
            thermo.coeffs_hi[n, 4]/4 * tc[4] + thermo.coeffs_hi[n, 5]/5 * tc[5] + (thermo.coeffs_hi[n, 6] - thermo.coeffs_lo[n, 6])
        end
    end

    return
end

# J/(m^3 K)
@inline function CV(T, rhoi, thermo, cp)
    tc = SVector{4, Float64}(T, T^2, T^3, T^4)

    cp_specs(cp, tc, thermo)
    result = 0
    for n = 1:Nspecs
        result += (cp[n] - 1)*rhoi[n]/thermo.mw[n]
    end
    return result*thermo.Ru
end

# J/(m^3 K)
@inline function CP(T, rhoi, thermo, cp)
    tc = SVector{4, Float64}(T, T^2, T^3, T^4)

    cp_specs(cp, tc, thermo)
    result = 0
    for n = 1:Nspecs
        result += cpi[n]*rhoi[n]/thermo.mw[n]
    end
    return result*thermo.Ru
end

# get mean internal energy in volume unit
# J/m^3
@inline function InternalEnergy(T, rhoi, thermo, ei)
    tc = SVector{5, Float64}(T, T^2, T^3, T^4, T^5)

    ei_specs(ei, tc, thermo)
    result = 0
    for n = 1:Nspecs
        result += rhoi[n] * ei[n]/thermo.mw[n]
    end
    return result * thermo.Ru
end

# get temperature from ρi and internal energy
@inline function GetT(ein, ρi, thermo)
    maxiter::Int64 = 100
    tol::Float64 = 1e-7
    tmin::Float64 = 250;  # max lower bound for thermo def
    tmax::Float64 = 3500; # min upper bound for thermo def

    tmp = MVector{Nspecs, Float64}(undef)
    emin = InternalEnergy(tmin, ρi, thermo, tmp)
    emax = InternalEnergy(tmax, ρi, thermo, tmp)
    if ein < emin
      # Linear Extrapolation below tmin
      cv = CV(tmin, ρi, thermo, tmp)
      T = tmin - (emin - ein) / cv
      return T
    end

    if ein > emax
      # Linear Extrapolation above tmax
      cv = CV(tmax, ρi, thermo, tmp)
      T = tmax - (emax - ein) / cv
      return T
    end
  
    As=0
    for n = 1:Nspecs
        As += (thermo.coeffs_lo[n, 1]-1) *thermo.Ru/thermo.mw[n]*ρi[n]
    end
  
    # initial value
    t1 = ein/As

    if t1 < tmin || t1 > tmax
        t1 = tmin + (tmax - tmin) / (emax - emin) * (ein - emin);
    end
  
    for i ∈ 1:maxiter
        e1 = InternalEnergy(t1, ρi, thermo, tmp)
        cv = CV(t1, ρi, thermo, tmp)

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

function initThermo(mech, Nspecs)
    ct = pyimport("cantera")
    gas = ct.Solution(mech)

    Ru = ct.gas_constant * 1e-3
    mw = gas.molecular_weights * 1e-3
    coeffs_sep = zeros(Float64, Nspecs)
    coeffs_hi = zeros(Float64, Nspecs, 7)
    coeffs_lo = zeros(Float64, Nspecs, 7)
    viscosity_poly = zeros(Float64, Nspecs, 5)
    conductivity_poly = zeros(Float64, Nspecs, 5)
    binarydiffusion_poly = zeros(Float64, Nspecs, Nspecs, 5)

    for i ∈ 1:Nspecs
        coeffs_sep[i] = gas.species(i-1).thermo.coeffs[1]
        coeffs_hi[i, :] = gas.species(i-1).thermo.coeffs[2:8]
        coeffs_lo[i, :] = gas.species(i-1).thermo.coeffs[9:end]
        viscosity_poly[i, :] = gas.get_viscosity_polynomial(i-1)
        conductivity_poly[i, :] = gas.get_thermal_conductivity_polynomial(i-1)
        for j ∈ 1:Nspecs
            binarydiffusion_poly[i, j, :] = gas.get_binary_diff_coeffs_polynomial(i-1, j-1)
        end
    end

    thermo = thermoProperty(Nspecs, Ru, CuArray(mw), CuArray(coeffs_sep), CuArray(coeffs_lo), 
                            CuArray(coeffs_hi), CuArray(viscosity_poly), 
                            CuArray(conductivity_poly), CuArray(binarydiffusion_poly))
    return thermo
end


# mech = "./air.yaml"
# ct = pyimport("cantera")
# gas = ct.Solution(mech)
# T::Float64 = 300
# P::Float64 = 0.025 * ct.one_atm
# gas.TPX = T, P, "N2:78 O2:21 AR:1"

# ρi = gas.Y * gas.density
# ρi_d = CuArray(ρi)
# tmp = CUDA.zeros(Float64, 8)
# thermo = initThermo(mech, 8)
# @cuda threads=16 GetT(6100, ρi_d, thermo, ρi_d)