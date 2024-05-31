# GPU chemical reaction
function eval_gpu(U, Q, ρi, dt, thermo, react)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    @inbounds T::Float32 = Q[i, j, k, 6]

    if T > T_criteria
        sc = MVector{Nspecs, Float32}(undef)
        wdot = @MVector zeros(Float32, Nspecs)
        @inbounds rho = @view ρi[i, j, k, :]

        for n = 1:Nspecs
            @inbounds sc[n] = rho[n]/thermo.mw[n] * 1f-6
        end
        
        vproductionRate(wdot, sc, T, thermo, react)

        Δei::Float32 = 0
        for n = 1:Nspecs
            @inbounds Δρ = wdot[n] * thermo.mw[n] * 1f6 * dt
            @inbounds Δei += -thermo.coeffs_lo[6, n] *  Δρ * thermo.Ru / thermo.mw[n]
            @inbounds rho[n] += Δρ
        end

        @inbounds U[i, j, k, 5] += Δei

        # update primitives
        @inbounds ρ = max(Q[i, j, k, 1], eps(Float32))
        ∑ρ::Float32 = 0
        for n = 1:Nspecs
            @inbounds rho[n] = max(rho[n], 0.f0)
            @inbounds ∑ρ += rho[n]
        end
        for n = 1:Nspecs
            @inbounds rho[n] *= ρ/∑ρ
        end
        # @inbounds rho[Nspecs] += ρ - ∑ρ

        @inbounds ein = Q[i, j, k, 7]
        ein += Δei
        T = max(GetT(ein, rho, thermo), eps(Float32))
        p = max(Pmixture(T, rho, thermo), eps(Float32))
        @inbounds Q[i, j, k, 5] = p
        @inbounds Q[i, j, k, 6] = T
        @inbounds Q[i, j, k, 7] = ein
    end
    return
end

# For stiff reaction, point implicit
function eval_gpu_stiff(U, Q, ρi, dt, thermo, react)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+NG || j > Nyp+NG || k > Nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    @inbounds T = Q[i, j, k, 6]

    if T > T_criteria
        sc = MVector{Nspecs, Float32}(undef)
        Δρ = MVector{Nspecs, Float32}(undef)
        Δρn = MVector{Nspecs, Float32}(undef)
        wdot = @MVector zeros(Float32, Nspecs)
        Arate = @MMatrix zeros(Float32, Nspecs, Nspecs)
        A1 = MMatrix{Nspecs, Nspecs, Float32}(undef)
        @inbounds rho = @view ρi[i, j, k, :]

        for n = 1:Nspecs
            @inbounds sc[n] = rho[n]/thermo.mw[n] * 1f-6
        end
        
        vproductionRate_Jac(wdot, sc, Arate, T, thermo, react)

        # I - AⁿΔt
        for l = 1:Nspecs
            for n = 1:Nspecs
                @inbounds A1[n, l] = (n == l ? 1.f0 : 0.f0) - 
                                    Arate[n, l] * thermo.mw[n] / thermo.mw[l] * dt
            end
        end

        for n = 1:Nspecs
            @inbounds Δρ[n] = wdot[n] * thermo.mw[n] * 1f6 * dt
        end

        # solve(x, A, b): Ax=b
        solve(Δρn, A1, Δρ)

        Δei::Float32 = 0
        for n = 1:Nspecs
            @inbounds Δei += -thermo.coeffs_lo[n, 6] *  Δρn[n] * thermo.Ru / thermo.mw[n]
            @inbounds rho[n] += Δρn[n]
        end

        @inbounds U[i, j, k, 5] += Δei

        # update primitives
        @inbounds ρ = max(Q[i, j, k, 1], eps(Float32))
        ∑ρ::Float32 = 0
        for n = 1:Nspecs
            @inbounds rho[n] = max(rho[n], 0.f0)
            @inbounds ∑ρ += rho[n]
        end
        for n = 1:Nspecs
            @inbounds rho[n] *= ρ/∑ρ
        end
        # @inbounds rho[Nspecs] += ρ - ∑ρ

        @inbounds ein::Float32 = Q[i, j, k, 7]
        ein += Δei
        T = max(GetT(ein, rho, thermo), eps(Float32))
        p = max(Pmixture(T, rho, thermo), eps(Float32))
        @inbounds Q[i, j, k, 5] = p
        @inbounds Q[i, j, k, 6] = T
        @inbounds Q[i, j, k, 7] = ein
    end
    return
end

@inline function vproductionRate(wdot, sc, T, thermo, react)
    gi_T = MVector{Nspecs, Float64}(undef)
    k_f_s = MVector{Nreacs, Float64}(undef)
    Kc_s = MVector{Nreacs, Float64}(undef)

    lgT::Float32 = log(T)
    invT::Float32 = 1.f0 / T
  
    @inbounds @fastmath for n = 1:Nreacs
        k_f_s[n] = react.Arr[1,n] * exp(react.Arr[2,n] * lgT - react.Arr[3,n] * invT)
    end
  
    # compute the Gibbs free energy 
    gibbs(gi_T, T, lgT, invT, thermo)
  
    RsT::Float64 = thermo.Ru / react.atm * 1e6 * T
  
    for n = 1:Nreacs
        Δgi::Float64 = 0
        for m = 1:Nspecs
            @inbounds Δgi += gi_T[m] * (react.vf[m, n]-react.vr[m, n])
        end
        @inbounds Kc_s[n] = RsT ^ react.sgm[n] * @fastmath(exp(Δgi))
    end
  
    for n = 1:Nreacs
        @inbounds q_f = k_f_s[n]
        @inbounds q_r = q_f / Kc_s[n]
        for m = 1:Nspecs
            @inbounds vf = react.vf[m, n]
            @inbounds vr = react.vr[m, n]
            if vf != 0
                @inbounds q_f *= sc[m] ^ vf
            end
            if vr != 0
                @inbounds q_r *= sc[m] ^ vr
            end
        end

        mixture::Float64 = 0
        # three body reaction
        if react.reaction_type[n] == 2
            for m = 1:Nspecs
                @inbounds mixture += react.ef[m, n] * sc[m]
            end
            q_f *= mixture
            q_r *= mixture
        elseif react.reaction_type[n] == 3
            for m = 1:Nspecs
                @inbounds mixture += react.ef[m, n] * sc[m]
            end

            @inbounds @fastmath redP = mixture / k_f_s[n] * react.loP[1,n] * exp(react.loP[2,n] * lgT - react.loP[3,n] * invT)
            @fastmath @fastmath logPred = log10(redP)
            @inbounds A = react.Troe[1,n]
            @inbounds @fastmath logFcent = log10((1-A)*exp(-T/react.Troe[2,n]) + A*exp(-T/react.Troe[3,n]) + exp(-react.Troe[4,n]*invT))
            troe_c = -0.4 - 0.67 * logFcent
            troe_n = 0.75 - 1.27 * logFcent
            troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred))
            F_troe = 10 ^ (logFcent / (1 + troe * troe))
            k_1 = (redP / (1 + redP)) * F_troe
            q_f *= k_1
            q_r *= k_1
        end

        for m = 1:Nspecs
            @inbounds wdot[m] += (q_f - q_r) * (react.vr[m, n] - react.vf[m, n])
        end
    end
    return
end

@inline function vproductionRate_Jac(wdot, sc, Arate, T, thermo, react)
    gi_T = MVector{Nspecs, Float64}(undef)
    k_f_s = MVector{Nreacs, Float64}(undef)
    Kc_s = MVector{Nreacs, Float64}(undef)

    lgT::Float32 = log(T)
    invT::Float32 = 1.f0 / T
  
    @inbounds @fastmath for n = 1:Nreacs
        k_f_s[n] = react.Arr[1,n] * exp(react.Arr[2,n] * lgT - react.Arr[3,n] * invT)
    end
  
    # compute the Gibbs free energy 
    gibbs(gi_T, T, lgT, invT, thermo)
  
    RsT::Float64 = thermo.Ru / react.atm * 1e6 * T
  
    for n = 1:Nreacs
        Δgi::Float64 = 0
        for m = 1:Nspecs
            @inbounds Δgi += gi_T[m] * (react.vf[m, n]-react.vr[m, n])
        end
        @inbounds Kc_s[n] = RsT ^ react.sgm[n] * @fastmath(exp(Δgi))
    end
  
    for n = 1:Nreacs
        @inbounds q_f = k_f_s[n]
        @inbounds q_r = q_f / Kc_s[n]
        for m = 1:Nspecs
            @inbounds vf = react.vf[m, n]
            @inbounds vr = react.vr[m, n]
            if vf != 0
                @inbounds q_f *= sc[m] ^ vf
            end
            if vr != 0
                @inbounds q_r *= sc[m] ^ vr
            end
        end

        mixture::Float64 = 0
        # three body reaction
        if react.reaction_type[n] == 2
            for m = 1:Nspecs
                @inbounds mixture += react.ef[m, n] * sc[m]
            end
            q_f *= mixture
            q_r *= mixture
        elseif react.reaction_type[n] == 3
            for m = 1:Nspecs
                @inbounds mixture += react.ef[m, n] * sc[m]
            end

            @inbounds @fastmath redP = mixture / k_f_s[n] * react.loP[1,n] * exp(react.loP[2,n] * lgT - react.loP[3,n] * invT)
            @fastmath logPred = log10(redP)
            @inbounds A = react.Troe[1,n]
            @inbounds @fastmath logFcent = log10((1-A)*exp(-T/react.Troe[2,n]) + A*exp(-T/react.Troe[3,n]) + exp(-react.Troe[4,n]*invT))
            troe_c = -0.4 - 0.67 * logFcent
            troe_n = 0.75 - 1.27 * logFcent
            troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred))
            F_troe = 10 ^ (logFcent / (1 + troe * troe))
            k_1 = (redP / (1 + redP)) * F_troe
            q_f *= k_1
            q_r *= k_1
        end

        for m = 1:Nspecs
            @inbounds wdot[m] += (q_f - q_r) * (react.vr[m, n] - react.vf[m, n])

            @inbounds invsc::Float64 = 1/(sc[m] + eps(Float64))
            @inbounds Awf = react.vf[m, n] * q_f * invsc
            @inbounds Awr = react.vr[m, n] * q_r * invsc
            for l = 1:Nspecs
                @inbounds Arate[l, m] += (Awf - Awr) * (react.vr[l, n] - react.vf[l, n])
            end
        end
    end
    return
end

# Solve Ax = b with Gauss Elimination
@inline function solve(x, a, b)
	@inbounds for k = 1:Nspecs-1
        # find main element      
        fmax = abs(a[k,k])
        kk = k
        @inbounds for i = k+1:Nspecs
            absa = abs(a[i, k])
            if absa > fmax
                fmax = absa
                kk = i
            end
        end
        # exchange line   
        @inbounds for j = k:Nspecs
            t = a[k,j]
            a[k,j] = a[kk,j]
            a[kk,j] = t
        end
        t = b[k]
        b[k] = b[kk]
        b[kk] = t
        # Elimination
        @inbounds for i = k+1:Nspecs
            f = -a[i,k]/a[k,k]
            @inbounds for j = 1:Nspecs
                a[i,j] += f*a[k,j]
            end
            b[i] += f*b[k]
        end
	end

    # get x    
	@inbounds x[Nspecs] = b[Nspecs]/a[Nspecs,Nspecs]
	@inbounds for i = Nspecs-1:-1:1
        x[i] = b[i]
        @inbounds for j = i+1:Nspecs
            x[i] -= a[i,j]*x[j]
        end
        x[i] /= a[i,i]
    end
end

# column major, in Float64
function initReact(mech)
    ct = pyimport("cantera")
    gas = ct.Solution(mech)

    atm::Float64 = ct.one_atm

    spec_names = gas.species_names
    reactant_stoich::Matrix{Int64} = gas.reactant_stoich_coeffs3
    product_stoich::Matrix{Int64} = gas.product_stoich_coeffs3
    reaction_type = zeros(Int64, Nreacs)
    delta_order = zeros(Int64, Nreacs)
    Arrhenius_rate = zeros(Float64, 3, Nreacs)
    low_rate = zeros(Float64, 3, Nreacs)
    Troe_coeffs = zeros(Float64, 4, Nreacs)
    efficiencies = zeros(Float64, Nspecs, Nreacs)

    for j = 1:Nreacs
        reaction_i = gas.reaction(j-1)

        if reaction_i.reaction_type == "Arrhenius"
            order = sum(reactant_stoich[:,j])
            reaction_type[j] = 1
            Arrhenius_rate[1, j] = reaction_i.rate.pre_exponential_factor * 1e3 ^ (order-1) # A, [m, mol, s]
            Arrhenius_rate[2, j] = reaction_i.rate.temperature_exponent # b
            Arrhenius_rate[3, j] = reaction_i.rate.activation_energy/ct.gas_constant # E, K
        elseif reaction_i.reaction_type == "three-body-Arrhenius"
            order = sum(reactant_stoich[:,j]) + 1
            reaction_type[j] = 2
            for i = 1:Nspecs
                efficiencies[i, j] = reaction_i.third_body.efficiency(spec_names[i])
            end
            Arrhenius_rate[1, j] = reaction_i.rate.pre_exponential_factor * 1e3 ^ (order-1) # A, [m, mol, s]
            Arrhenius_rate[2, j] = reaction_i.rate.temperature_exponent # b
            Arrhenius_rate[3, j] = reaction_i.rate.activation_energy/ct.gas_constant # E, K
        elseif reaction_i.reaction_type == "falloff-Troe"
            order = sum(reactant_stoich[:,j])
            reaction_type[j] = 3
            for i = 1:Nspecs
                efficiencies[i, j] = reaction_i.third_body.efficiency(spec_names[i])
            end
            Arrhenius_rate[1, j] = reaction_i.rate.high_rate.pre_exponential_factor * 1e3 ^ (order-1) # A, [m, mol, s]
            Arrhenius_rate[2, j] = reaction_i.rate.high_rate.temperature_exponent # b
            Arrhenius_rate[3, j] = reaction_i.rate.high_rate.activation_energy/ct.gas_constant # E, K
            low_rate[1, j] = reaction_i.rate.low_rate.pre_exponential_factor * 1e3 ^ (order) # A, [m, mol, s]
            low_rate[2, j] = reaction_i.rate.low_rate.temperature_exponent # b
            low_rate[3, j] = reaction_i.rate.low_rate.activation_energy/ct.gas_constant # E, K
            len = length(reaction_i.rate.falloff_coeffs)
            if len == 3
                Troe_coeffs[1:3, j] = reaction_i.rate.falloff_coeffs
                Troe_coeffs[4, j] = 1e30
            elseif len == 4
                Troe_coeffs[:, j] = reaction_i.rate.falloff_coeffs
            else
                error("Not correct Troe coefficients in reaction $j")
            end
        else
            error("Not supported reaction type of reaction $j")
        end

        d_order::Int64 = 0
        for i = 1:Nspecs
            d_order += reactant_stoich[i,j] - product_stoich[i,j]
        end
        delta_order[j] = d_order
    end
    react = reactionProperty(atm, ROCArray(reaction_type), ROCArray(delta_order), 
                             ROCArray(reactant_stoich), ROCArray(product_stoich), 
                             ROCArray(Arrhenius_rate), ROCArray(efficiencies), 
                             ROCArray(low_rate), ROCArray(Troe_coeffs))
    return react
end