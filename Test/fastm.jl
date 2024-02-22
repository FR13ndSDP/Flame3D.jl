using CUDA, BenchmarkTools
using CUDA:i32

function m1(a, x::Float64)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    b = 1.0
    sum = 0.0
    for n = 1:100
        b = (Float64(n)+0.1)
        b = b*b
        b = b*b*b
        sum += b
    end
    a[i] = sum
    return
end

function m2(a, x::Float64)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    b = 1.0
    sum = 0.0
    for n = 1:100
        b = @fastmath (Float64(n)+0.1)^6
        sum += b
    end
    a[i] = sum
    return
end


a = CUDA.zeros(Float64, 800)

@benchmark CUDA.@sync @cuda fastmath=true threads=800 m1(a, 0.12)