using CUDA, BenchmarkTools, Plots

# initialize
const N::Int64 = 1024
ϕ = ones(Float64, N, N)
ϕ[1, :] .= 10.0

function evolve(ϕ, ϕn, N)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y

    if i < 2 || i > N-1 || j < 2 || j > N-1
        return
    end

    @inbounds ϕn[i,j] = 0.25*(ϕ[i-1,j] + ϕ[i+1,j] + ϕ[i,j-1] + ϕ[i,j+1])
    return
end

function evolve_shared(ϕ, ϕn, N)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    
    if i > N || j > N
        return
    end

    s = CuStaticSharedArray(Float64, 18*18)
    local_i = threadIdx().x
    local_j = threadIdx().y

    @inbounds s[local_i * 18 + local_j+1] = ϕ[i, j]
    #top 
    if local_i == 1 && i != 1
        @inbounds s[local_j+1] = ϕ[i-1, j]
    end
    #bottom
    if local_i == 16 && i != N
        @inbounds s[17*18+local_j+1] = ϕ[i+1, j]
    end
    #left
    if local_j == 1 && j != 1
        @inbounds s[local_i*18+local_j] = ϕ[i, j-1]
    end
    #right
    if local_j == 16 && j != N
        @inbounds s[local_i*18+local_j+2] = ϕ[i, j+1]
    end

    sync_threads()

    if i < 2 || i > N-1 || j < 2 || j > N-1
        return
    end

    @inbounds ϕn[i,j] = 0.25*(s[(local_i-1)*18+local_j+1] + s[(local_i+1)*18+local_j+1] 
                             +s[local_i*18+local_j] + s[local_i*18+local_j+2])
    return
end

function run(ϕ, N)
    nthreads = (16, 16)
    nblock = (cld(N, 16), cld(N, 16))

    ϕ_d = CuArray(ϕ)
    ϕn_d = copy(ϕ_d)

    for _ ∈ 1:5000
        @cuda blocks=nblock threads=nthreads evolve(ϕ_d, ϕn_d, N)
        ϕ_d, ϕn_d = ϕn_d, ϕ_d
    end

    copyto!(ϕ, ϕ_d)
end

run(ϕ, N)
heatmap(ϕ)