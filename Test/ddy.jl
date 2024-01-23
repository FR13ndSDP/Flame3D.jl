using PyCall

plt = pyimport("matplotlib.pyplot")
a = zeros(Float64, 128)

# for i = 1:128
#     a[i] = tanh((i-64)/128*12π)
# end
a[1:8] .= 0.0178995
a[9:end] .= 0.0274578
plt.plot(a, label="Initial")

dt = 5e-6
k = 0.0007712910310550843 * ones(Float64, 128)
for i = 1:128
    k[i] *= 1+rand(Float64)
end
dx = 0.000158

b = copy(a)
f = copy(a)

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
plt.plot(a, "--", label="half - avg2")

a = zeros(Float64, 128)
a[1:8] .= 0.0178995
a[9:end] .= 0.0274578
b = copy(a)
for _ = 1:2000
    for i = 3:127
        f[i] = (k[i]+k[i-1])/2 * (15/12*b[i]-15/12*b[i-1] + 1/12*b[i-2]-1/12*b[i+1])/dx
    end

    for i = 3:126
        a[i] = b[i] + dt * (f[i+1]-f[i])/dx
    end
    a[1:8] .= 0.0178995
    b .= a
end
plt.plot(a, "--", label="half - avg4")


a = zeros(Float64, 128)
a[1:8] .= 0.0178995
a[9:end] .= 0.0274578
b = copy(a)
for _ = 1:2000
    for i = 2:127
        f[i] = (k[i]) * (b[i+1]-b[i-1])/(2*dx)
    end

    for i = 3:126
        a[i] = b[i] + dt * (f[i+1]-f[i-1])/(2*dx)
    end
    a[1:8] .= 0.0178995
    b .= a
end
plt.plot(a, label="two times - 2nd")

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
plt.plot(a, label="two times - 4th")

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
plt.plot(a, label="two times - 6th")

plt.legend()
plt.show()
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
