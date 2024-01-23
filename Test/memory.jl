using CUDA

# @inline function callit(a)
#     a[1] = 1
#     a[2] = 2
#     a[3] = 3
#     return
# end

# function kernel(a)
#     i = (blockIdx().x-1)* blockDim().x + threadIdx().x
#     callit(@view a[i,:])
#     return
# end

function test(a)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    b = CuStaticSharedArray(Float64, 7)

    b[1] = i
    b[2] = i*2
    b[3] = i*3
    b[4] = i*4
    b[5] = i*5
    b[6] = i*6
    b[7] = i*7
    @inbounds a[i] = sum(@view b[1:7])
    return
end

function test2(a, b)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x

    @inbounds b[i, 1] = i
    @inbounds b[i, 2] = i*2
    @inbounds b[i, 3] = i*3
    @inbounds b[i, 4] = i*4
    @inbounds b[i, 5] = i*5
    @inbounds b[i, 6] = i*6
    @inbounds b[i, 7] = i*7
    @inbounds a[i] = sum(@view b[i, 1:7])
    return
end

a1 = zeros(Float64, 128)
a2 = zeros(Float64, 128)
a = CuArray(zeros(Float64, 128))
b = CuArray(zeros(Float64, 128, 7))
CUDA.@time @cuda threads=128 test(a)
copyto!(a1, a)
CUDA.@time @cuda threads=128 test2(a, b)
copyto!(a2, a)