# using CUDA

# # @inline function callit(a)
# #     a[1] = 1
# #     a[2] = 2
# #     a[3] = 3
# #     return
# # end

# # function kernel(a)
# #     i = (blockIdx().x-1)* blockDim().x + threadIdx().x
# #     callit(@view a[i,:])
# #     return
# # end

# function test(a)
#     i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    
#     a[i] = 128.6
#     b = (CUDA.sqrt(a[i]) - sqrt(a[i]))
#     @cuprintln("$b")
#     return
# end

# a_h = zeros(Float64, 128)

# a = cu(a_h)
# CUDA.@time @cuda threads=128 test(a)

using Plots

a = zeros(Float64, 128)

# for i = 1:128
#     a[i] = tanh((i-64)/128*12π)
# end
a[1:8] .= 0.0178995
a[9:end] .= 0.0274578
# plot(a)

dt = 1e-5
k = 0.0007712910310550843 * ones(Float64, 128)
# for i = 1:128
#     k[i] *= rand(Float64)
# end
dx = 0.000158

b = copy(a)
f = copy(a)
p = plot()

for _ = 1:2000
    for i = 3:127
        f[i] = (k[i]+k[i-1])/2 * (b[i]-b[i-1])/dx
    end

    for i = 3:126
        a[i] = b[i] + dt * (f[i+1]-f[i])/dx
    end
    a[1:8] .= 0.0178995
    b .= a
end
plot!(p, a, lab="m1")

a = zeros(Float64, 128)
a[1:8] .= 0.0178995
a[9:end] .= 0.0274578
b = copy(a)
for _ = 1:2000
    for i = 3:127
        f[i] = (k[i]+k[i-1] + k[i-2]+k[i+1])/4 * (b[i]-b[i-1])/dx
    end

    for i = 3:126
        a[i] = b[i] + dt * (f[i+1]-f[i])/dx
    end
    a[1:8] .= 0.0178995
    b .= a
end
plot!(p, a, lab="m2")

a = zeros(Float64, 128)
a[1:8] .= 0.0178995
a[9:end] .= 0.0274578
b = copy(a)
for _ = 1:2000
    for i = 2:127
        f[i] = k[i] * (b[i+1]-b[i-1])/(2*dx)
    end

    for i = 3:126
        a[i] = b[i] + dt * (f[i+1]-f[i-1])/(2*dx)
    end
    a[1:8] .= 0.0178995
    b .= a
end
plot!(p, a, lab="m3")

a = zeros(Float64, 128)
a[1:8] .= 0.0178995
a[9:end] .= 0.0274578
b = copy(a)
for _ = 1:2000
    for i = 3:126
        f[i] = k[i] * (2/3*b[i+1]-2/3*b[i-1] + 1/12*b[i-2] -1/12*b[i+2])/(dx)
    end

    for i = 5:124
        a[i] = b[i] + dt * (2/3*f[i+1]-2/3*f[i-1] + 1/12*f[i-2] - 1/12*f[i+2])/(dx)
    end
    a[1:8] .= 0.0178995
    b .= a
end
plot!(p, a, lab="m4")

a = zeros(Float64, 128)
a[1:8] .= 0.0178995
a[9:end] .= 0.0274578
b = copy(a)
for _ = 1:2000
    for i = 4:125
        f[i] = k[i] * (3/4*b[i+1]-3/4*b[i-1] + 3/20*b[i-2] -3/20*b[i+2] +1/60*b[i+3]-1/60*b[i-3])/(dx)
    end

    for i = 7:122
        a[i] = b[i] + dt * (3/4*f[i+1]-3/4*f[i-1] + 3/20*f[i-2] - 3/20*f[i+2] + 1/60*f[i+3]-1/60*f[i-3])/(dx)
    end
    a[1:8] .= 0.0178995
    b .= a
end
plot!(p, a, lab="m5")

# a = zeros(Float64, 128)
# a[1:8] .= 0.0178995
# a[9:end] .= 0.0274578
# b = copy(a)

# for _ = 1:20000
#     for i = 3:126
#         a[i] = b[i] + dt/dx^2 * k[i] * (b[i+1]+b[i-1]-2*b[i])
#     end
#     a[1:8] .= 0.0178995
#     b .= a
# end
# plot!(p, a, lab="2th")
