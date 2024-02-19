using PyCall

plt = pyimport("matplotlib.pyplot")
a = zeros(Float64, 128)

# for i = 1:128
#     a[i] = tanh((i-64)/128*12π)
# end
a .= 300
a[60:68] .= 2000
# plt.plot(a, label="Initial")

dt = 1e-4
ρ = 10.0
k = 0.0355 * ones(Float64, 128)/(ρ * 1004.5)
for i = 1:128
    k[i] *= 1+rand(Float64)
end
dx = 0.0003

b = copy(a)
f = copy(a)

for _ = 1:2000
    for i = 3:127
        f[i] = (k[i]+k[i-1])/2 * (b[i]-b[i-1])/dx
    end

    for i = 3:126
        a[i] = b[i] + dt * (f[i+1]-f[i])/dx
    end
    a[1:8] .= 300
    b .= a
end
plt.plot(a, "-*", label="half - 2nd order")

a = zeros(Float64, 128)
a .= 300
a[60:68] .= 2000
b = copy(a)
for _ = 1:2000
    for i = 3:127
        f[i] = (k[i]+k[i-1])/2 * (15/12*b[i]-15/12*b[i-1] + 1/12*b[i-2]-1/12*b[i+1])/dx
    end

    for i = 3:126
        a[i] = b[i] + dt * (f[i+1]-f[i])/dx
    end
    a[1:8] .= 300
    b .= a
end
plt.plot(a, "-+", label="half - 4th order")

a = zeros(Float64, 128)
a .= 300
a[60:68] .= 2000
b = copy(a)
for _ = 1:2000
    for i = 4:126
        f[i] = (k[i]+k[i-1])/2 * (49/36*(b[i]-b[i-1]) - 5/36*(b[i+1]-b[i-2]) + 1/90*(b[i+2]-b[i-3]))/dx
    end

    for i = 4:125
        a[i] = b[i] + dt * (f[i+1]-f[i])/dx
    end
    a[1:8] .= 300
    b .= a
end
plt.plot(a, "-o", label="half - 6th order")

a = zeros(Float64, 128)
a .= 300
a[60:68] .= 2000
b = copy(a)
for tt = 1:2000
    for i = 2:127
        f[i] = (k[i]) * (b[i+1]-b[i-1])/(2*dx)
    end

    for i = 3:126
        a[i] = b[i] + dt * (f[i+1]-f[i-1])/(2*dx)
    end
    a[1:8] .= 300
    b .= a

    # filtering, cannot handle discontinuity
    if tt %1 == 0
        for i = 5:124
            a[i] = 93/128*(b[i]+b[i])/2 + 7/16*(b[i-1]+b[i+1])/2 - 7/32*(b[i-2]+b[i+2])/2 + 1/16*(b[i-3]+b[i+3])/2 -
            1/128*(b[i-4]+b[i+4])/2 
        end
        b .= a
    end
end
plt.plot(a, label="two times - 2nd order", "--")

a = zeros(Float64, 128)
a .= 300
a[60:68] .= 2000
b = copy(a)
for tt = 1:2000
    for i = 3:126
        f[i] = k[i] * (2/3*b[i+1]-2/3*b[i-1] + 1/12*b[i-2] -1/12*b[i+2])/(dx)
    end

    for i = 5:124
        a[i] = b[i] + dt * (2/3*f[i+1]-2/3*f[i-1] + 1/12*f[i-2] - 1/12*f[i+2])/(dx)
    end
    a[1:8] .= 300
    b .= a

    # filtering, cannot handle discontinuity
    if tt %1 == 0
        for i = 5:124
            a[i] = 93/128*(b[i]+b[i])/2 + 7/16*(b[i-1]+b[i+1])/2 - 7/32*(b[i-2]+b[i+2])/2 + 1/16*(b[i-3]+b[i+3])/2 -
            1/128*(b[i-4]+b[i+4])/2 
        end
        b .= a
    end
end
plt.plot(a, label="two times - 4th order", "--")

a = zeros(Float64, 128)
a .= 300
a[60:68] .= 2000
b = copy(a)
for tt = 1:2000
    for i = 4:125
        f[i] = k[i] * (3/4*b[i+1]-3/4*b[i-1] + 3/20*b[i-2] -3/20*b[i+2] +1/60*b[i+3]-1/60*b[i-3])/(dx)
    end

    for i = 7:122
        a[i] = b[i] + dt * (3/4*f[i+1]-3/4*f[i-1] + 3/20*f[i-2] - 3/20*f[i+2] + 1/60*f[i+3]-1/60*f[i-3])/(dx)
    end
    a[1:8] .= 300
    b .= a

    # filtering, cannot handle discontinuity
    if tt %1 == 0
        for i = 5:124
            a[i] = 93/128*(b[i]+b[i])/2 + 7/16*(b[i-1]+b[i+1])/2 - 7/32*(b[i-2]+b[i+2])/2 + 1/16*(b[i-3]+b[i+3])/2 -
            1/128*(b[i-4]+b[i+4])/2 
        end
        b .= a
    end
end

plt.plot(a, label="two times - 6th order", "--")

a = zeros(Float64, 128)
a .= 300
a[60:68] .= 2000
b = copy(a)
for _ = 1:2000
    for i = 3:126
        a[i] = b[i] + dt * ((k[i+1]-k[i-1])/(2*dx) * (b[i+1]-b[i-1])/(2*dx) + k[i] * (b[i-1]+b[i+1]-2*b[i])/(dx*dx))
    end

    a[1:8] .= 300
    b .= a
end
plt.plot(a, "-.", label="laplacian - 2nd order")

a = zeros(Float64, 128)
a .= 300
a[60:68] .= 2000
b = copy(a)
for _ = 1:2000
    for i = 3:126
        a[i] = b[i] + dt * ((1/12*k[i-2]-2/3*k[i-1]+2/3*k[i+1]-1/12*k[i+2])/(dx) * (1/12*b[i-2]-2/3*b[i-1]+2/3*b[i+1]-1/12*b[i+2])/(dx) + k[i] * (-1/12*b[i-2]+4/3*b[i-1]-5/2*b[i]+4/3*b[i+1]-1/12*b[i+2])/(dx*dx))
    end

    a[1:8] .= 300
    b .= a
end
plt.plot(a, "-.", label="laplacian - 4th order")

plt.title("non-constant heat diffusion")
plt.legend()
plt.show()
