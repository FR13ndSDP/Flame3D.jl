function shockSensor(ϕ, Q, Nx, Ny, Nz, NG)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i < 1+NG || i > Nx+NG || j < 1+NG || j > Ny+NG || k < 1+NG || k > Nz+NG
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
    ϕx = CUDA.abs(-Px1 + 2*Px2 - Px3)/(Px1 + 2*Px2 + Px3)
    ϕy = CUDA.abs(-Py1 + 2*Py2 - Py3)/(Py1 + 2*Py2 + Py3)
    ϕz = CUDA.abs(-Pz1 + 2*Pz2 - Pz3)/(Pz1 + 2*Pz2 + Pz3)
    ϕ[i, j, k] = ϕx + ϕy + ϕz
    return
end

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
function WENO_x(F, ϕ, Fp, Fm, NG, Nx, Ny, Nz, NV)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nx-1 || j > Ny-2 || k > Nz-2
        return
    end

    eps::Float64 = CUDA.eps(1e-16)
    tmp1::Float64 = 13/12
    tmp2::Float64 = 1/6

    c1::Float64 = 0.85/30
    c2::Float64 = -5.75/30
    c3::Float64 = 22/30
    c4::Float64 = 0.5
    c5::Float64 = -0.075
    c6::Float64 = 0.005

    # Jameson sensor
    ϕx = ϕ[i+1+NG, j+1+NG, k+1+NG] + ϕ[i+2+NG, j+1+NG, k+1+NG] + ϕ[i+NG, j+1+NG, k+1+NG]

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
    elseif ϕx < 0.8
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
    else
        for n = 1:NV
            @inbounds fp = Fp[i+NG, j+1+NG, k+1+NG, n] + 0.5*minmod(Fp[i+1+NG, j+1+NG, k+1+NG, n]-Fp[i+NG, j+1+NG, k+1+NG, n], Fp[i+NG, j+1+NG, k+1+NG, n] - Fp[i-1+NG, j+1+NG, k+1+NG, n])
            @inbounds fm = Fm[i+1+NG, j+1+NG, k+1+NG, n] - 0.5*minmod(Fm[i+2+NG, j+1+NG, k+1+NG, n]-Fm[i+1+NG, j+1+NG, k+1+NG, n], Fm[i+1+NG, j+1+NG, k+1+NG, n] - Fm[i+NG, j+1+NG, k+1+NG, n])
            @inbounds F[i, j, k, n] = fp + fm
        end
    end
    return
end

#Range: 1 -> N-1
function WENO_y(F, ϕ, Fp, Fm, NG, Nx, Ny, Nz, NV)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nx-2 || j > Ny-1 || k > Nz-2
        return
    end

    eps::Float64 = CUDA.eps(1e-16)
    tmp1::Float64 = 13/12
    tmp2::Float64 = 1/6

    c1::Float64 = 0.85/30
    c2::Float64 = -5.75/30
    c3::Float64 = 22/30
    c4::Float64 = 0.5
    c5::Float64 = -0.075
    c6::Float64 = 0.005

    # Jameson sensor
    ϕy = ϕ[i+1+NG, j+1+NG, k+1+NG] + ϕ[i+1+NG, j+2+NG, k+1+NG] + ϕ[i+1+NG, j+NG, k+1+NG]

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
    elseif ϕy < 0.8
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
    else
        for n = 1:NV
            @inbounds fp = Fp[i+1+NG, j+NG, k+1+NG, n] + 0.5*minmod(Fp[i+1+NG, j+1+NG, k+1+NG, n]-Fp[i+1+NG, j+NG, k+1+NG, n], Fp[i+1+NG, j+NG, k+1+NG, n] - Fp[i+1+NG, j-1+NG, k+1+NG, n])
            @inbounds fm = Fm[i+1+NG, j+1+NG, k+1+NG, n] - 0.5*minmod(Fm[i+1+NG, j+2+NG, k+1+NG, n]-Fm[i+1+NG, j+1+NG, k+1+NG, n], Fm[i+1+NG, j+1+NG, k+1+NG, n] - Fm[i+1+NG, j+NG, k+1+NG, n])
            @inbounds F[i, j, k, n] = fp + fm
        end  
    end
    return
end

#Range: 1 -> N-1
function WENO_z(F, ϕ, Fp, Fm, NG, Nx, Ny, Nz, NV)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nx-2 || j > Ny-2 || k > Nz-1
        return
    end

    eps::Float64 = CUDA.eps(1e-16)
    tmp1::Float64 = 13/12
    tmp2::Float64 = 1/6

    c1::Float64 = 0.85/30
    c2::Float64 = -5.75/30
    c3::Float64 = 22/30
    c4::Float64 = 0.5
    c5::Float64 = -0.075
    c6::Float64 = 0.005

    # Jameson sensor
    ϕz = ϕ[i+1+NG, j+1+NG, k+1+NG] + ϕ[i+1+NG, j+1+NG, k+2+NG] + ϕ[i+1+NG, j+1+NG, k+NG]

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
    elseif ϕz < 0.8
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
    else
        for n = 1:NV
            @inbounds fp = Fp[i+1+NG, j+1+NG, k+NG, n] + 0.5*minmod(Fp[i+1+NG, j+1+NG, k+1+NG, n]-Fp[i+1+NG, j+1+NG, k+NG, n], Fp[i+1+NG, j+1+NG, k+NG, n] - Fp[i+1+NG, j+1+NG, k-1+NG, n])
            @inbounds fm = Fm[i+1+NG, j+1+NG, k+1+NG, n] - 0.5*minmod(Fm[i+1+NG, j+1+NG, k+2+NG, n]-Fm[i+1+NG, j+1+NG, k+1+NG, n], Fm[i+1+NG, j+1+NG, k+1+NG, n] - Fm[i+1+NG, j+1+NG, k+NG, n])
            @inbounds F[i, j, k, n] = fp + fm
        end    
    end
    return
end