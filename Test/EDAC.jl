using WriteVTK

const N = 256
const Re = 1000.f0
const M = 0.1f0
const L = 1.f0
const dx = L/(N-1)
const ϵ = 1/M^2

"""
    ΔtCFL = (1/Δx + 1/Δy + 1/Ma*sqrt(1/Δx^2 + 1/Δy^2))^-1 
    Δtdiff = Re/2(1/Δx^2 + 1/Δy^2)^-1
"""
const ΔtCFL = 1/(2/dx + 1/M*sqrt(2/dx^2)) 
const Δtdiff = Re/2/(2/dx^2)
const dt = min(ΔtCFL, Δtdiff)

@show dt ϵ Re


function step(Nstep)
    Q = zeros(Float32, 3, N, N) # P, u, v
    Qn = zeros(Float32, 3, N, N) # P, u, v
    F = zeros(Float32, 3, N, N)
    G = zeros(Float32, 3, N, N)
    S = zeros(Float32, 3, N-2, N-2)

    # write result coordinate
    x = zeros(Float32, N, N)
    y = zeros(Float32, N, N)

    for j = 1:N
        x[:, j] = collect(0:dx:L)
    end

    for i = 1:N
        y[i, :] = collect(0:dx:L)
    end

    # set lid velocity: 1.0
    Q[2, :, N] .= 1.0

    # iter
    for tt = 1:Nstep
        # Fx
        @inbounds @simd for j = 1:N
            for i = 1:N
                F[1, i, j] = Q[2, i, j]*(Q[1, i, j] + ϵ)
                F[2, i, j] = Q[2, i, j]^2 + Q[1, i, j]
                F[3, i, j] = Q[2, i, j]*Q[3, i, j]
            end
        end

        # Fy
        @inbounds @simd for j = 1:N
            for i = 1:N
                G[1, i, j] = Q[3, i, j]*(Q[1, i, j] + ϵ)
                G[2, i, j] = Q[3, i, j]*Q[2, i, j]
                G[3, i, j] = Q[3, i, j]^2 + Q[1, i, j]
            end
        end

        # S
        @inbounds @simd for j = 1:N-2
            for i = 1:N-2
                for n = 1:3
                    S[n, i, j] = 1/Re * ((Q[n, i, j+1] + Q[n, i+2, j+1] - 2*Q[n, i+1, j+1])/dx^2 + 
                                        (Q[n, i+1, j] + Q[n, i+1, j+2] - 2*Q[n, i+1, j+1])/dx^2)
                end
            end
        end

        copyto!(Qn, Q)

        # predictor
        @inbounds @simd for j = 2:N-1
            for i = 2:N-1
                for n = 1:3
                    Q[n, i, j] -= dt/dx * (F[n, i+1, j] - F[n, i, j] +
                                           G[n, i, j+1] - G[n, i, j])

                    Q[n, i, j] += S[n, i-1, j-1] * dt
                end
            end
        end

        # Fx
        @inbounds @simd for j = 1:N
            for i = 1:N
                F[1, i, j] = Q[2, i, j]*(Q[1, i, j] + ϵ)
                F[2, i, j] = Q[2, i, j]^2 + Q[1, i, j]
                F[3, i, j] = Q[2, i, j]*Q[3, i, j]
            end
        end

        # Fy
        @inbounds @simd for j = 1:N
            for i = 1:N
                G[1, i, j] = Q[3, i, j]*(Q[1, i, j] + ϵ)
                G[2, i, j] = Q[3, i, j]*Q[2, i, j]
                G[3, i, j] = Q[3, i, j]^2 + Q[1, i, j]
            end
        end

        # S
        @inbounds @simd for j = 1:N-2
            for i = 1:N-2
                for n = 1:3
                    S[n, i, j] = 1/Re * ((Q[n, i, j+1] + Q[n, i+2, j+1] - 2*Q[n, i+1, j+1])/dx^2 + 
                                         (Q[n, i+1, j] + Q[n, i+1, j+2] - 2*Q[n, i+1, j+1])/dx^2)
                end
            end
        end

        # corrector
        @inbounds @simd for j = 2:N-1
            for i = 2:N-1
                for n = 1:3
                    Q[n, i, j] = 0.5*(Qn[n, i, j] + Q[n, i, j] - dt/dx*(F[n, i, j] - F[n, i-1, j]) - dt/dx*(G[n, i, j] - G[n, i, j-1]) + dt * S[n, i-1, j-1])
                end
            end
        end

        # P: zero gradient
        @views @. Q[1, 1, :] = (2*Q[1, 2, :] - 0.5*Q[1, 3, :])/1.5
        @views @. Q[1, N, :] = (2*Q[1, N-1, :] - 0.5*Q[1, N-2, :])/1.5
        @views @. Q[1, :, 1] = (2*Q[1, :, 2] - 0.5*Q[1, :, 3])/1.5
        @views @. Q[1, :, N] = (2*Q[1, :, N-1] - 0.5*Q[1, :, N-2])/1.5

        if tt % 10000 == 0
            vtk_grid(string("Out", tt, ".vts"), x, y) do f
                f["p"] = @view Q[1, :, :]
                f["velocity"] = @view Q[2:3, :, :]
            end
        end
    end

    return nothing
end

@time step(100000)