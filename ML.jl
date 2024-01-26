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

    @inbounds T = Q[i+NG, j+NG, k+NG, 6]
    @inbounds P = Q[i+NG, j+NG, k+NG, 5]
    @inbounds rhoi = @view ρi[i+NG, j+NG, k+NG, :]

    # only T > 2000 K calculate reaction
    if T > 3000 && P < 10132.5
        @inbounds T1 = T + yt_pred[Nspecs+1, i + Nxp*(j-1 + Ny*(k-1))] * dt
        @inbounds U[i+NG, j+NG, k+NG, 5] += InternalEnergy(T1, rhoi, thermo) - InternalEnergy(T, rhoi, thermo)
        for n = 1:Nspecs
            @inbounds Yi = (lambda * (yt_pred[n, i + Nxp*(j-1 + Ny*(k-1))] * dt + inputs[n+2, i + Nxp*(j-1 + Ny*(k-1))]) + 1) ^ (1/lambda)
            @inbounds ρi[i+NG, j+NG, k+NG, n] = Yi * Q[i+NG, j+NG, k+NG, 1]
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