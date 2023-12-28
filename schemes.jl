@inline function minmod(a, b)
    ifelse(a*b > 0, (CUDA.abs(a) > CUDA.abs(b)) ? b : a, zero(a))
end

#Range: 1 -> N-1
function NND_x(F, Fp, Fm, NG, Nx, Ny, Nz, NV)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nx-1 || j > Ny-2 || k > Nz-2
        return
    end
    for n = 1:NV
        @inbounds fp = Fp[i+NG, j+1+NG, k+1+NG, n] + 0.5*minmod(Fp[i+1+NG, j+1+NG, k+1+NG, n]-Fp[i+NG, j+1+NG, k+1+NG, n], Fp[i+NG, j+1+NG, k+1+NG, n] - Fp[i-1+NG, j+1+NG, k+1+NG, n])
        @inbounds fm = Fm[i+1+NG, j+1+NG, k+1+NG, n] - 0.5*minmod(Fm[i+2+NG, j+1+NG, k+1+NG, n]-Fm[i+1+NG, j+1+NG, k+1+NG, n], Fm[i+1+NG, j+1+NG, k+1+NG, n] - Fm[i+NG, j+1+NG, k+1+NG, n])
        @inbounds F[i, j, k, n] = fp + fm
    end
    return
end

function NND_y(F, Fp, Fm, NG, Nx, Ny, Nz, NV)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nx-2 || j > Ny-1 || k > Nz-2
        return
    end
    for n = 1:NV
        @inbounds fp = Fp[i+1+NG, j+NG, k+1+NG, n] + 0.5*minmod(Fp[i+1+NG, j+1+NG, k+1+NG, n]-Fp[i+1+NG, j+NG, k+1+NG, n], Fp[i+1+NG, j+NG, k+1+NG, n] - Fp[i+1+NG, j-1+NG, k+1+NG, n])
        @inbounds fm = Fm[i+1+NG, j+1+NG, k+1+NG, n] - 0.5*minmod(Fm[i+1+NG, j+2+NG, k+1+NG, n]-Fm[i+1+NG, j+1+NG, k+1+NG, n], Fm[i+1+NG, j+1+NG, k+1+NG, n] - Fm[i+1+NG, j+NG, k+1+NG, n])
        @inbounds F[i, j, k, n] = fp + fm
    end
    return
end

function NND_z(F, Fp, Fm, NG, Nx, Ny, Nz, NV)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nx-2 || j > Ny-2 || k > Nz-1
        return
    end
    for n = 1:NV
        @inbounds fp = Fp[i+1+NG, j+1+NG, k+NG, n] + 0.5*minmod(Fp[i+1+NG, j+1+NG, k+1+NG, n]-Fp[i+1+NG, j+1+NG, k+NG, n], Fp[i+1+NG, j+1+NG, k+NG, n] - Fp[i+1+NG, j+1+NG, k-1+NG, n])
        @inbounds fm = Fm[i+1+NG, j+1+NG, k+1+NG, n] - 0.5*minmod(Fm[i+1+NG, j+1+NG, k+2+NG, n]-Fm[i+1+NG, j+1+NG, k+1+NG, n], Fm[i+1+NG, j+1+NG, k+1+NG, n] - Fm[i+1+NG, j+1+NG, k+NG, n])
        @inbounds F[i, j, k, n] = fp + fm
    end
    return
end

#Range: 1 -> N-1
function WENO_x(F, Q, Fp, Fm, NG, Nx, Ny, Nz, NV, ϕ)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nx-1 || j > Ny-2 || k > Nz-2
        return
    end

    eps::Float64 = CUDA.eps(1e-10)
    tmp1::Float64 = 13/12
    tmp2::Float64 = 1/6

    c1::Float64 = 0.85/30
    c2::Float64 = -5.75/30
    c3::Float64 = 22/30
    c4::Float64 = 0.5
    c5::Float64 = -0.075
    c6::Float64 = 0.005

    # Jameson sensor
    @inbounds P1 = Q[i+NG, j+1+NG, k+1+NG, 5]
    @inbounds P2 = Q[i+1+NG, j+1+NG, k+1+NG, 5]
    @inbounds P3 = Q[i+2+NG, j+1+NG, k+1+NG, 5]
    ϕx = CUDA.abs(-P1 + 2*P2 - P3)/(P1 + 2*P2 + P3)

    ϕ[i+1+NG, j+1+NG, k+1+NG, 1] = ϕx

    if ϕx < 0.02
        for n = 1:NV
            @inbounds V1 = Fp[i-2+NG, j+1+NG, k+1+NG, n]
            @inbounds V2 = Fp[i-1+NG, j+1+NG, k+1+NG, n]
            @inbounds V3 = Fp[i+NG,   j+1+NG, k+1+NG, n]
            @inbounds V4 = Fp[i+1+NG, j+1+NG, k+1+NG, n]
            @inbounds V5 = Fp[i+2+NG, j+1+NG, k+1+NG, n]
            @inbounds V6 = Fp[i+3+NG, j+1+NG, k+1+NG, n]
            
            fpx = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6

            @inbounds V1 = Fm[i-2+NG, j+1+NG, k+1+NG, n]
            @inbounds V2 = Fm[i-1+NG, j+1+NG, k+1+NG, n]
            @inbounds V3 = Fm[i+NG,   j+1+NG, k+1+NG, n]
            @inbounds V4 = Fm[i+1+NG, j+1+NG, k+1+NG, n]
            @inbounds V5 = Fm[i+2+NG, j+1+NG, k+1+NG, n]
            @inbounds V6 = Fm[i+3+NG, j+1+NG, k+1+NG, n]

            fmx = c6*V1 + c5*V2 + c4*V3 + c3*V4 + c2*V5 + c1*V6

            @inbounds F[i, j, k, n] = fpx + fmx
        end
    else
        for n = 1:NV
            @inbounds V1 = Fp[i-2+NG, j+1+NG, k+1+NG, n]
            @inbounds V2 = Fp[i-1+NG, j+1+NG, k+1+NG, n]
            @inbounds V3 = Fp[i+NG,   j+1+NG, k+1+NG, n]
            @inbounds V4 = Fp[i+1+NG, j+1+NG, k+1+NG, n]
            @inbounds V5 = Fp[i+2+NG, j+1+NG, k+1+NG, n]
            # FP
            s11 = tmp1*(V1-2*V2+V3)^2 + 0.25*(V1-4*V2+3*V3)^2
            s22 = tmp1*(V2-2*V3+V4)^2 + 0.25*(V2-V4)^2
            s33 = tmp1*(V3-2*V4+V5)^2 + 0.25*(3*V3-4*V4+V5)^2

            s11 = 1/(eps+s11)^2
            s22 = 1/(eps+s22)^2
            s33 = 1/(eps+s33)^2
            # τ = CUDA.abs(s11-s33)
            # s11 = 1 + (τ/(eps+s11))^2
            # s22 = 1 + (τ/(eps+s22))^2
            # s33 = 1 + (τ/(eps+s33))^2

            a1 = s11
            a2 = 6*s22
            a3 = 3*s33
            invsum = 1/(a1+a2+a3)

            v1 = 2*V1-7*V2+11*V3
            v2 = -V2+5*V3+2*V4
            v3 = 2*V3+5*V4-V5
            fpx = tmp2*invsum*(a1*v1+a2*v2+a3*v3)

            @inbounds V1 = Fm[i-1+NG, j+1+NG, k+1+NG, n]
            @inbounds V2 = Fm[i+NG,   j+1+NG, k+1+NG, n]
            @inbounds V3 = Fm[i+1+NG, j+1+NG, k+1+NG, n]
            @inbounds V4 = Fm[i+2+NG, j+1+NG, k+1+NG, n]
            @inbounds V5 = Fm[i+3+NG, j+1+NG, k+1+NG, n]
            # FM
            s11 = tmp1*(V5-2*V4+V3)^2 + 0.25*(V5-4*V4+3*V3)^2
            s22 = tmp1*(V2-2*V3+V4)^2 + 0.25*(V4-V2)^2
            s33 = tmp1*(V3-2*V2+V1)^2 + 0.25*(3*V3-4*V2+V1)^2

            s11 = 1/(eps+s11)^2
            s22 = 1/(eps+s22)^2
            s33 = 1/(eps+s33)^2
            # τ = CUDA.abs(s11-s33)
            # s11 = 1 + (τ/(eps+s11))^2
            # s22 = 1 + (τ/(eps+s22))^2
            # s33 = 1 + (τ/(eps+s33))^2

            a1 = s11
            a2 = 6*s22
            a3 = 3*s33
            invsum = 1/(a1+a2+a3)

            v1 = 11*V3-7*V4+2*V5
            v2 = -V4+5*V3+2*V2
            v3 = 2*V3+5*V2-V1
            fmx = tmp2*invsum*(a1*v1+a2*v2+a3*v3)
            
            @inbounds F[i, j, k, n] = fpx + fmx
        end
    end
    return
end

#Range: 1 -> N-1
function WENO_y(F, Q, Fp, Fm, NG, Nx, Ny, Nz, NV, ϕ)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nx-2 || j > Ny-1 || k > Nz-2
        return
    end

    eps::Float64 = CUDA.eps(1e-10)
    tmp1::Float64 = 13/12
    tmp2::Float64 = 1/6

    c1::Float64 = 0.85/30
    c2::Float64 = -5.75/30
    c3::Float64 = 22/30
    c4::Float64 = 0.5
    c5::Float64 = -0.075
    c6::Float64 = 0.005

    # Jameson sensor
    @inbounds P1 = Q[i+1+NG, j+NG, k+1+NG, 5]
    @inbounds P2 = Q[i+1+NG, j+1+NG, k+1+NG, 5]
    @inbounds P3 = Q[i+1+NG, j+2+NG, k+1+NG, 5]
    ϕy = CUDA.abs(-P1 + 2*P2 - P3)/(P1 + 2*P2 + P3)

    ϕ[i+1+NG, j+1+NG, k+1+NG, 2] = ϕy

    if ϕy < 0.02
        for n = 1:NV
            @inbounds V1 = Fp[i+1+NG, j-2+NG, k+1+NG, n]
            @inbounds V2 = Fp[i+1+NG, j-1+NG, k+1+NG, n]
            @inbounds V3 = Fp[i+1+NG, j+NG,   k+1+NG, n]
            @inbounds V4 = Fp[i+1+NG, j+1+NG, k+1+NG, n]
            @inbounds V5 = Fp[i+1+NG, j+2+NG, k+1+NG, n]
            @inbounds V6 = Fp[i+1+NG, j+3+NG, k+1+NG, n]
            
            fpy = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6

            @inbounds V1 = Fm[i+1+NG, j-2+NG, k+1+NG, n]
            @inbounds V2 = Fm[i+1+NG, j-1+NG, k+1+NG, n]
            @inbounds V3 = Fm[i+1+NG, j+NG,   k+1+NG, n]
            @inbounds V4 = Fm[i+1+NG, j+1+NG, k+1+NG, n]
            @inbounds V5 = Fm[i+1+NG, j+2+NG, k+1+NG, n]
            @inbounds V6 = Fm[i+1+NG, j+3+NG, k+1+NG, n]

            fmy = c6*V1 + c5*V2 + c4*V3 + c3*V4 + c2*V5 + c1*V6

            @inbounds F[i, j, k, n] = fpy + fmy
        end
    else
        for n = 1:NV
            @inbounds V1 = Fp[i+1+NG, j-2+NG, k+1+NG, n]
            @inbounds V2 = Fp[i+1+NG, j-1+NG, k+1+NG, n]
            @inbounds V3 = Fp[i+1+NG, j+NG,   k+1+NG, n]
            @inbounds V4 = Fp[i+1+NG, j+1+NG, k+1+NG, n]
            @inbounds V5 = Fp[i+1+NG, j+2+NG, k+1+NG, n]
            # FP
            s11 = tmp1*(V1-2*V2+V3)^2 + 0.25*(V1-4*V2+3*V3)^2
            s22 = tmp1*(V2-2*V3+V4)^2 + 0.25*(V2-V4)^2
            s33 = tmp1*(V3-2*V4+V5)^2 + 0.25*(3*V3-4*V4+V5)^2

            s11 = 1/(eps+s11)^2
            s22 = 1/(eps+s22)^2
            s33 = 1/(eps+s33)^2
            # τ = CUDA.abs(s11-s33)
            # s11 = 1 + (τ/(eps+s11))^2
            # s22 = 1 + (τ/(eps+s22))^2
            # s33 = 1 + (τ/(eps+s33))^2

            a1 = s11
            a2 = 6*s22
            a3 = 3*s33
            invsum = 1/(a1+a2+a3)

            v1 = 2*V1-7*V2+11*V3
            v2 = -V2+5*V3+2*V4
            v3 = 2*V3+5*V4-V5
            fpy = tmp2*invsum*(a1*v1+a2*v2+a3*v3)

            @inbounds V1 = Fm[i+1+NG, j-1+NG, k+1+NG, n]
            @inbounds V2 = Fm[i+1+NG, j+NG,   k+1+NG, n]
            @inbounds V3 = Fm[i+1+NG, j+1+NG, k+1+NG, n]
            @inbounds V4 = Fm[i+1+NG, j+2+NG, k+1+NG, n]
            @inbounds V5 = Fm[i+1+NG, j+3+NG, k+1+NG, n]
            # FM
            s11 = tmp1*(V5-2*V4+V3)^2 + 0.25*(V5-4*V4+3*V3)^2
            s22 = tmp1*(V2-2*V3+V4)^2 + 0.25*(V4-V2)^2
            s33 = tmp1*(V3-2*V2+V1)^2 + 0.25*(3*V3-4*V2+V1)^2

            s11 = 1/(eps+s11)^2
            s22 = 1/(eps+s22)^2
            s33 = 1/(eps+s33)^2
            # τ = CUDA.abs(s11-s33)
            # s11 = 1 + (τ/(eps+s11))^2
            # s22 = 1 + (τ/(eps+s22))^2
            # s33 = 1 + (τ/(eps+s33))^2

            a1 = s11
            a2 = 6*s22
            a3 = 3*s33
            invsum = 1/(a1+a2+a3)

            v1 = 11*V3-7*V4+2*V5
            v2 = -V4+5*V3+2*V2
            v3 = 2*V3+5*V2-V1
            fmy = tmp2*invsum*(a1*v1+a2*v2+a3*v3)
            
            @inbounds F[i, j, k, n] = fpy + fmy
        end
    end
    return
end

#Range: 1 -> N-1
function WENO_z(F, Q, Fp, Fm, NG, Nx, Ny, Nz, NV, ϕ)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nx-2 || j > Ny-2 || k > Nz-1
        return
    end

    eps::Float64 = CUDA.eps(1e-10)
    tmp1::Float64 = 13/12
    tmp2::Float64 = 1/6

    c1::Float64 = 0.85/30
    c2::Float64 = -5.75/30
    c3::Float64 = 22/30
    c4::Float64 = 0.5
    c5::Float64 = -0.075
    c6::Float64 = 0.005

    # Jameson sensor
    @inbounds P1 = Q[i+1+NG, j+1+NG, k+NG, 5]
    @inbounds P2 = Q[i+1+NG, j+1+NG, k+1+NG, 5]
    @inbounds P3 = Q[i+1+NG, j+1+NG, k+2+NG, 5]
    ϕz = CUDA.abs(-P1 + 2*P2 - P3)/(P1 + 2*P2 + P3)

    ϕ[i+1+NG, j+1+NG, k+1+NG, 3] = ϕz

    if ϕz < 0.02
        for n = 1:NV
            @inbounds V1 = Fp[i+1+NG, j+1+NG, k-2+NG, n]
            @inbounds V2 = Fp[i+1+NG, j+1+NG, k-1+NG, n]
            @inbounds V3 = Fp[i+1+NG, j+1+NG, k+NG,   n]
            @inbounds V4 = Fp[i+1+NG, j+1+NG, k+1+NG, n]
            @inbounds V5 = Fp[i+1+NG, j+1+NG, k+2+NG, n]
            @inbounds V6 = Fp[i+1+NG, j+1+NG, k+3+NG, n]
            
            fpz = c1*V1 + c2*V2 + c3*V3 + c4*V4 + c5*V5 + c6*V6

            @inbounds V1 = Fm[i+1+NG, j+1+NG, k-2+NG, n]
            @inbounds V2 = Fm[i+1+NG, j+1+NG, k-1+NG, n]
            @inbounds V3 = Fm[i+1+NG, j+1+NG, k+NG,   n]
            @inbounds V4 = Fm[i+1+NG, j+1+NG, k+1+NG, n]
            @inbounds V5 = Fm[i+1+NG, j+1+NG, k+2+NG, n]
            @inbounds V6 = Fm[i+1+NG, j+1+NG, k+3+NG, n]

            fmz = c6*V1 + c5*V2 + c4*V3 + c3*V4 + c2*V5 + c1*V6

            @inbounds F[i, j, k, n] = fpz + fmz
        end
    else
        for n = 1:NV
            @inbounds V1 = Fp[i+1+NG, j+1+NG, k-2+NG, n]
            @inbounds V2 = Fp[i+1+NG, j+1+NG, k-1+NG, n]
            @inbounds V3 = Fp[i+1+NG, j+1+NG, k+NG,   n]
            @inbounds V4 = Fp[i+1+NG, j+1+NG, k+1+NG, n]
            @inbounds V5 = Fp[i+1+NG, j+1+NG, k+2+NG, n]
            # FP
            s11 = tmp1*(V1-2*V2+V3)^2 + 0.25*(V1-4*V2+3*V3)^2
            s22 = tmp1*(V2-2*V3+V4)^2 + 0.25*(V2-V4)^2
            s33 = tmp1*(V3-2*V4+V5)^2 + 0.25*(3*V3-4*V4+V5)^2

            s11 = 1/(eps+s11)^2
            s22 = 1/(eps+s22)^2
            s33 = 1/(eps+s33)^2
            # τ = CUDA.abs(s11-s33)
            # s11 = 1 + (τ/(eps+s11))^2
            # s22 = 1 + (τ/(eps+s22))^2
            # s33 = 1 + (τ/(eps+s33))^2

            a1 = s11
            a2 = 6*s22
            a3 = 3*s33
            invsum = 1/(a1+a2+a3)

            v1 = 2*V1-7*V2+11*V3
            v2 = -V2+5*V3+2*V4
            v3 = 2*V3+5*V4-V5
            fpy = tmp2*invsum*(a1*v1+a2*v2+a3*v3)

            @inbounds V1 = Fm[i+1+NG, j+1+NG, k-1+NG, n]
            @inbounds V2 = Fm[i+1+NG, j+1+NG, k+NG,   n]
            @inbounds V3 = Fm[i+1+NG, j+1+NG, k+1+NG, n]
            @inbounds V4 = Fm[i+1+NG, j+1+NG, k+2+NG, n]
            @inbounds V5 = Fm[i+1+NG, j+1+NG, k+3+NG, n]
            # FM
            s11 = tmp1*(V5-2*V4+V3)^2 + 0.25*(V5-4*V4+3*V3)^2
            s22 = tmp1*(V2-2*V3+V4)^2 + 0.25*(V4-V2)^2
            s33 = tmp1*(V3-2*V2+V1)^2 + 0.25*(3*V3-4*V2+V1)^2

            s11 = 1/(eps+s11)^2
            s22 = 1/(eps+s22)^2
            s33 = 1/(eps+s33)^2
            # τ = CUDA.abs(s11-s33)
            # s11 = 1 + (τ/(eps+s11))^2
            # s22 = 1 + (τ/(eps+s22))^2
            # s33 = 1 + (τ/(eps+s33))^2

            a1 = s11
            a2 = 6*s22
            a3 = 3*s33
            invsum = 1/(a1+a2+a3)

            v1 = 11*V3-7*V4+2*V5
            v2 = -V4+5*V3+2*V2
            v3 = 2*V3+5*V2-V1
            fmy = tmp2*invsum*(a1*v1+a2*v2+a3*v3)
            
            @inbounds F[i, j, k, n] = fpy + fmy
        end
    end
    return
end