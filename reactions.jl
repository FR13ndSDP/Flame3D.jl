# Collect input
function pre_input(inputs, inputs_norm, Q, Y, lambda, inputs_mean, inputs_std)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp || j > Ny || k > Nz
        return
    end

    @inbounds inputs[1, i + Nxp*(j-1 + Ny*(k-1))] = Q[i+NG, j+NG, k+NG, 6] # T
    @inbounds inputs[2, i + Nxp*(j-1 + Ny*(k-1))] = Q[i+NG, j+NG, k+NG, 5] # p

    for n = 3:Nspecs+2
        @inbounds Yi = Y[i+NG, j+NG, k+NG, n-2]
        @inbounds inputs[n, i + Nxp*(j-1 + Ny*(k-1))] = (Yi^lambda - 1) / lambda
    end

    for n = 1:Nspecs+2
        @inbounds inputs_norm[n, i + Nxp*(j-1 + Ny*(k-1))] = (inputs[n, i + Nxp*(j-1 + Ny*(k-1))] - inputs_mean[n]) / inputs_std[n]
    end
    return
end

# Parse prediction
function post_predict(yt_pred, inputs, U, Q, ρi, dt, lambda, thermo)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp || j > Ny || k > Nz
        return
    end

    @inbounds ρ = Q[i+NG, j+NG, k+NG, 1]
    @inbounds T = Q[i+NG, j+NG, k+NG, 6]
    @inbounds P = Q[i+NG, j+NG, k+NG, 5]
    @inbounds rhoi = @view ρi[i+NG, j+NG, k+NG, :]
    ρnew = MVector{Nspecs, Float64}(undef)

    # only T > 2000 K calculate reaction
    if T > 3000 && P < 10132.5
        @inbounds T1 = T + yt_pred[Nspecs+1, i + Nxp*(j-1 + Ny*(k-1))] * dt

        for n = 1:Nspecs
            @inbounds Yi = (lambda * (yt_pred[n, i + Nxp*(j-1 + Ny*(k-1))] * dt + inputs[n+2, i + Nxp*(j-1 + Ny*(k-1))]) + 1) ^ (1/lambda)
            @inbounds ρnew[n] = Yi * ρ
        end

        @inbounds U[i+NG, j+NG, k+NG, 5] += InternalEnergy(T1, ρnew, thermo) - InternalEnergy(T, rhoi, thermo)
        for n = 1:Nspecs
            ρi[i+NG, j+NG, k+NG, n] = ρnew[n]
        end
    end
    return
end

# Zero GPU allocation
function evalModel(Y1, Y2, output, w1, w2, w3, b1, b2, b3, input)
    mul!(Y1, w1, input)
    Y1 .+= b1
    @. Y1 = gelu(Y1)

    mul!(Y2, w2, Y1)
    Y2 .+= b2
    @. Y2 = gelu(Y2)

    mul!(output, w3, Y2)
    output .+= b3

    return
end

# Collect input for CPU evaluation (1 D)
function pre_input_cpu(inputs, Q, Y)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp || j > Ny || k > Nz
        return
    end

    @inbounds inputs[1, i + Nxp*(j-1 + Ny*(k-1))] = Q[i+NG, j+NG, k+NG, 6] # T
    @inbounds inputs[2, i + Nxp*(j-1 + Ny*(k-1))] = Q[i+NG, j+NG, k+NG, 5] # p

    for n = 3:Nspecs+2
        @inbounds inputs[n, i + Nxp*(j-1 + Ny*(k-1))] = Y[i+NG, j+NG, k+NG, n-2]
    end

    return
end

# Parse output for CPU evaluation (1 D)
function post_eval_cpu(yt_pred, U, Q, ρi, thermo)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp || j > Ny || k > Nz
        return
    end
    
    @inbounds ρ = Q[i+NG, j+NG, k+NG, 1]
    @inbounds T = Q[i+NG, j+NG, k+NG, 6]
    @inbounds rhoi = @view ρi[i+NG, j+NG, k+NG, :]
    ρnew = MVector{Nspecs, Float64}(undef)

    for n = 1:Nspecs
        @inbounds Yi = yt_pred[n, i + Nxp*(j-1 + Ny*(k-1))]
        @inbounds ρnew[n] = Yi * ρ
    end

    @inbounds T1::Float64 = yt_pred[Nspecs+1, i + Nxp*(j-1 + Ny*(k-1))]
    @inbounds U[i+NG, j+NG, k+NG, 5] += InternalEnergy(T1, ρnew, thermo) - InternalEnergy(T, rhoi, thermo)

    for n = 1:Nspecs
        @inbounds ρi[i+NG, j+NG, k+NG, n] = ρnew[n]
    end
    
    return
end

# serial on cpu using cantera: tooooooo slow
function eval_cpu(outputs, inputs, dt)
    ct = pyimport("cantera")
    gas = ct.Solution(mech)

    for i = 1:Nxp*Ny*Nz
        @inbounds T = inputs[1, i]
        @inbounds P = inputs[2, i]
        @inbounds Yi = @view inputs[3:end, i]
        gas.TPY = T, P, Yi
        r = ct.IdealGasReactor(gas)
        sim = ct.ReactorNet([r])
        sim.advance(dt)
        outputs[1:Nspecs,i] = gas.Y
        outputs[end, i] = gas.T
    end
end

# GPU chemical reaction
function eval_gpu(U, Q, ρi, dt, thermo)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Ny+NG || k > Nz+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    sc = MVector{Nspecs, Float64}(undef)
    wdot = @MVector zeros(Float64, Nspecs)
    @inbounds T = Q[i, j, k, 6]

    for n = 1:Nspecs
        @inbounds sc[n] = ρi[i, j, k, n]/thermo.mw[n] * 1e-6
    end
    
    vproductionRate(wdot, sc, T, thermo)

    Δei::Float64 = 0
    for n = 1:Nspecs
        @inbounds Δρ = wdot[n] * thermo.mw[n] * 1e6 * dt
        @inbounds Δei += -thermo.coeffs_lo[n, 6] *  Δρ * thermo.Ru / thermo.mw[n]
        @inbounds ρi[i, j, k, n] += Δρ
    end

    @inbounds U[i, j, k, 5] += Δei
    return
end

# For stiff reaction, point implicit
function eval_gpu_stiff(U, Q, ρi, dt, thermo)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Ny+NG || k > Nz+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    sc = MVector{Nspecs, Float64}(undef)
    Δρ = MVector{Nspecs, Float64}(undef)
    Δρn = MVector{Nspecs, Float64}(undef)
    wdot = @MVector zeros(Float64, Nspecs)
    Arate = @MMatrix zeros(Float64, Nspecs, Nspecs)
    A1 = MMatrix{Nspecs, Nspecs, Float64}(undef)
    @inbounds T = Q[i, j, k, 6]

    for n = 1:Nspecs
        @inbounds sc[n] = ρi[i, j, k, n]/thermo.mw[n] * 1e-6
    end
    
    vproductionRate_Jac(wdot, sc, Arate, T, thermo)

    # I - AⁿΔt
    for n = 1:Nspecs
        for l = 1:Nspecs
            @inbounds A1[n, l] = (n == l ? 1.0 : 0.0) - 
                                 Arate[n, l] * thermo.mw[n] / thermo.mw[l] * dt
        end
    end

    for n = 1:Nspecs
        @inbounds Δρ[n] = wdot[n] * thermo.mw[n] * 1e6 * dt
    end

    # solve(x, A, b): Ax=b
    solve(Δρn, A1, Δρ)

    Δei::Float64 = 0
    for n = 1:Nspecs
        @inbounds Δei += -thermo.coeffs_lo[n, 6] *  Δρn[n] * thermo.Ru / thermo.mw[n]
        @inbounds ρi[i, j, k, n] += Δρ[n]
    end

    @inbounds U[i, j, k, 5] += Δei
    return
end

# TODO: reaction rate for air.yaml, make it more general
# O, O2, N, NO, N2
@inline function vproductionRate(wdot, sc, T, thermo)
    gi_T = MVector{Nspecs, Float64}(undef)
    k_f_s = MVector{Nreacs, Float64}(undef)
    Kc_s = MVector{Nreacs, Float64}(undef)
    q_f = MVector{Nreacs, Float64}(undef)
    q_r = MVector{Nreacs, Float64}(undef)
    vf = @MMatrix zeros(Int64, Nspecs, Nreacs)
    vr = @MMatrix zeros(Int64, Nspecs, Nreacs)

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
  
    RsT::Float64 = thermo.Ru / thermo.atm * 1e6 * T
  
    Kc_s[1] = 1.0/RsT * exp(gi_T[5]- 2 * gi_T[3])
    Kc_s[2] = 1.0/RsT * exp(gi_T[2]- 2 * gi_T[1])
    Kc_s[3] = 1.0/RsT * exp(gi_T[4]- (gi_T[1] + gi_T[3]))
    Kc_s[4] = exp((gi_T[1] + gi_T[5]) - (gi_T[3] + gi_T[4]))
    Kc_s[5] = exp((gi_T[1] + gi_T[4]) - (gi_T[2] + gi_T[3]))
  
    mixture::Float64 = 0.0
  
    for n = 1:Nspecs
        @inbounds mixture += sc[n]
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

    for m = 1:Nreacs
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
    k_f_s = MVector{Nreacs, Float64}(undef)
    Kc_s = MVector{Nreacs, Float64}(undef)
    q_f = MVector{Nreacs, Float64}(undef)
    q_r = MVector{Nreacs, Float64}(undef)
    vf = @MMatrix zeros(Int64, Nspecs, Nreacs)
    vr = @MMatrix zeros(Int64, Nspecs, Nreacs)

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
  
    RsT::Float64 = thermo.Ru / thermo.atm * 1e6 * T
  
    Kc_s[1] = 1.0/RsT * exp(gi_T[5]- 2 * gi_T[3])
    Kc_s[2] = 1.0/RsT * exp(gi_T[2]- 2 * gi_T[1])
    Kc_s[3] = 1.0/RsT * exp(gi_T[4]- (gi_T[1] + gi_T[3]))
    Kc_s[4] = exp((gi_T[1] + gi_T[5]) - (gi_T[3] + gi_T[4]))
    Kc_s[5] = exp((gi_T[1] + gi_T[4]) - (gi_T[2] + gi_T[3]))
  
    mixture::Float64 = 0.0
  
    for n = 1:Nspecs
        @inbounds mixture += sc[n]
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

    for m = 1:Nreacs
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