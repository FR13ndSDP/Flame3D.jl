# Collect input
function pre_input(inputs, inputs_norm, Q, Y, lambda, inputs_mean, inputs_std)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

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
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

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
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

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
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

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
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nxp+NG || j > Ny+NG || k > Nz+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    sc = MVector{Nspecs, Float64}(undef)
    wdot = MVector{Nspecs, Float64}(undef)
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