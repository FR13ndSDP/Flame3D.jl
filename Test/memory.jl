using CUDA
using StaticArrays

function test(a)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    b = MVector{20, Float64}(undef)
    for n = 1:20
        @inbounds b[n] = n
    end

    @inbounds a[i] = sum(b)
    return
end

function test2(a, b)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x

    for n = 1:20
        @inbounds b[n] = n
    end

    @inbounds a[i] = sum(b)
    return
end

a1 = zeros(Float64, 128)
a2 = zeros(Float64, 128)
a = CuArray(zeros(Float64, 128))
b = CuArray(zeros(Float64, 128, 20))
CUDA.@time @cuda threads=128 test(a)
copyto!(a1, a)
@show a1
CUDA.@time @cuda threads=128 test2(a, b)
copyto!(a2, a)