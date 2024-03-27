using ReadVTK
using PyCall
using DelimitedFiles

# using HDF5

# const NG::Int64 = h5read("../metrics.h5", "NG")

fname = "../PLT/avg0-50000.vts"

plt = pyimport("matplotlib.pyplot")
plt.rc("text", usetex= false)
plt.rc("font", family= "sans-serif")
# plt.rc("font", sans-serif = "Helvetica")
plt.rc("font", size=15)

vtk = VTKFile(fname)

# point data
p_data = get_point_data(vtk)

# mesh cordinate
x,y,z = get_coordinates(vtk)
Nx, Ny, Nz = size(x)


# variables
p = get_data_reshaped(p_data["p"])
u = get_data_reshaped(p_data["u"])
v = get_data_reshaped(p_data["v"])
T = get_data_reshaped(p_data["T"])
ρ = get_data_reshaped(p_data["rho"])

pw = p[:, 1, :]
p∞ = sum(p[1, Ny, :])/Nz

# average in spanwise
pw1d = zeros(Float32, Nx)

for i = 1:Nx
    pw1d[i] = sum(pw[i, :])/Nz/p∞
end

# find x=-30mm δ
x1d = @view x[:, 1, 1]
id = partialsortperm(abs.(x1d.+0.03), 1)

uy = zeros(Float32, Ny)
for j = 1:Ny
    uy[j] = sum(u[id, j, :])/Nz
end

u∞ = uy[Ny]
id2 = partialsortperm(abs.(uy.-u∞*0.99), 1)
δ = y[id, id2, 1]

# pw plot
dns = readdlm("../SCU-benchmark/dns-p.txt", ',', Float32)
plt.subplot(2, 2, 1)
plt.plot(x1d[1:Nx-20]./δ, pw1d[1:Nx-20])
plt.plot(dns[:, 1], dns[:, 2], "+")
plt.xlim([-10, 6])
plt.xlabel(raw"$x/\delta$")
plt.ylabel(raw"$p_w/p_{\infty}$")
plt.title("pw")
# plt.show()

# u plot at -30mm
dns = readdlm("../SCU-benchmark/dns-u.txt", ',', Float32)
plt.subplot(2,2,2)
yy = y[id, :, 1]./δ
plt.plot(yy, uy/u∞)
plt.plot(dns[:, 1], dns[:, 2], "+")
plt.xlim([0, 1])
plt.xlabel(raw"$y/\delta$")
plt.ylabel(raw"$U/U_{\infty}$")
plt.title("-30mm, delta=$δ")
# plt.show()

# # u plot at -20mm
# id = partialsortperm(abs.(x1d.+0.02), 1)
# uy = zeros(Float32, Ny)
# vy = zeros(Float32, Ny)
# for j = 1:Ny
#     uy[j] = sum(u[id, j, :])/Nz
#     vy[j] = sum(v[id, j, :])/Nz
# end

# u∞ = uy[Ny]

# plt.subplot(2,2,2)
# plt.plot(y[id, :, 1]./δ, sqrt.(uy.^2+vy.^2)/u∞)
# plt.show()

# Cf
Tw = 307.f0
μ = 1.458f-6*Tw*sqrt(Tw)/(Tw+110.4f0)

u1d = zeros(Float32, Nx)
v1d = zeros(Float32, Nx)
ρ1d = zeros(Float32, Nx)
for i = 1:Nx
    u1d[i] = sum(u[i, 2, :])/Nz
    v1d[i] = sum(v[i, 2, :])/Nz
    ρ1d[i] = sum(ρ[i, 2, :])/Nz
end
ρ∞ = ρ[1, Ny, 1]

cf= zeros(Float32, Nx)
idcorner = partialsortperm(abs.(x1d), 1)

for i = 1:idcorner
    cf[i] = μ*u1d[i]/(y[i, 2, 1]-y[i, 1, 1])/(0.5*ρ∞*u∞^2)
end

for i = idcorner+1:Nx
    cf[i] = μ*(u1d[i]*cos(24/180*π) + v1d[i]*sin(24/180*π))/((y[i, 2, 1]-y[i, 1, 1])/cos(24/180*π))/(0.5*ρ∞*u∞^2)
end

dns = readdlm("../SCU-benchmark/dns-cf.txt", ',', Float32)
plt.subplot(2,2,3)
plt.plot(x1d[1:Nx-20]*1000, cf[1:Nx-20])
plt.plot(dns[:, 1], dns[:, 2], "+")
plt.xlabel(raw"$x/mm$")
plt.ylabel(raw"$C_f$")
# plt.show()

# -60 mm y+
id = partialsortperm(abs.(x1d.+0.1), 1)

ρy = zeros(Float32, Ny)
nu = zeros(Float32, Ny)
uplus = zeros(Float32, Ny)
uplus_d = zeros(Float32, Ny)
yplus = zeros(Float32, Ny)

for j = 1:Ny
    uy[j] = sum(u[id, j, :])/Nz
    ρy[j] = sum(ρ[id, j, :])/Nz
    tmp = sum(T[id, j, :])/Nz
    nu[j] = (1.458f-6*tmp*sqrt(tmp)/(tmp+110.4f0))/ρy[j]
end

ρw = ρy[1]
τw = (ρy[1]*nu[1] + ρy[2]*nu[2])/2 * uy[2]/y[id, 2, 1]
for j = 1:Ny
    yplus[j] = y[id, j, 1]*sqrt(τw/ρy[j])/nu[j]
    uplus[j] = uy[j]/sqrt(τw/ρy[j])
end

# van driest transform
for j = 2:Ny
    for jj = 2:j
        uplus_d[j] += sqrt((ρy[jj]+ρy[jj-1])/2/ρw)*(uplus[jj]-uplus[jj-1])
    end
end

plt.subplot(2,2,4)
plt.plot(yplus, uplus)
plt.plot(yplus, yplus)
plt.plot(yplus, 1/0.41*log.(yplus).+6.0)
plt.legend([raw"$u^+$", raw"$y^+$", raw"$\frac{1}{0.41}\log y^++6$"])
plt.ylim([0, 35])
plt.xscale("log")
plt.xlabel(raw"$y^+$")
plt.ylabel(raw"$u^+_{vd}$")
plt.title("-100mm, u profile")
plt.tight_layout()

# schlieren
function CD6(f)
    fₓ = 1/60*(f[7]-f[1]) - 3/20*(f[6]-f[2]) + 3/4*(f[5]-f[3])
    return fₓ
end

function CD2_L(f)
    fₓ = 2*f[2] - 0.5*f[3] - 1.5*f[1]
    return fₓ
end

function CD2_R(f)
    fₓ = -2*f[2] + 0.5*f[1] + 1.5*f[3]
    return fₓ
end

# fid = h5open("../metrics.h5", "r")
# dξdx = fid["dξdx"][NG+1:Nx+NG, NG+1:Ny+NG, 1]
# dξdy = fid["dξdy"][NG+1:Nx+NG, NG+1:Ny+NG, 1]
# dηdx = fid["dηdx"][NG+1:Nx+NG, NG+1:Ny+NG, 1]
# dηdy = fid["dηdy"][NG+1:Nx+NG, NG+1:Ny+NG, 1]
# close(fid)

# ρ2d = @view ρ[:, :, 1]
# ρξ = similar(ρ2d)
# ρη = similar(ρ2d)
# ρx = similar(ρ2d)
# ρy = similar(ρ2d)
# ∇ρ = similar(ρ2d)

# for j = 1:Ny, i = 4:Nx-3
#     ρξ[i, j] = CD6(ρ2d[i-3:i+3, j]) 
# end

# for j = 1:Ny, i = 1:3
#     ρξ[i, j] = CD2_L(ρ2d[i:i+2, j])
# end

# for j = 1:Ny, i = Nx-2:Nx
#     ρξ[i, j] = CD2_R(ρ2d[i-2:i, j])
# end

# for j = 4:Ny-3, i = 1:Nx
#     ρη[i, j] = CD6(ρ2d[i, j-3:j+3])
# end

# for j = 1:3, i = 1:Nx
#     ρη[i, j] = CD2_L(ρ2d[i, j:j+2])
# end

# for j = Ny-2:Ny, i = 1:Nx
#     ρη[i, j] = CD2_R(ρ2d[i, j-2:j])
# end

# @. ρx = ρξ*dξdx + ρη*dηdx
# @. ρy = ρξ*dξdy + ρη*dηdy
# @. ∇ρ = sqrt(ρx^2+ρy^2)

# max∇ρ = maximum(∇ρ)
# min∇ρ = minimum(∇ρ)

# @. ∇ρ = 0.8*exp(-10*(∇ρ-max∇ρ)/(max∇ρ-min∇ρ))

plt.figure()
plt.contourf(x[:, :, 1], y[:, :, 1], T[:, :, 1], 100, cmap="coolwarm")
plt.axis("equal")
plt.show()
# fig = plt.figure(figsize=(18, 6))
# # lev= np.linspace(1e-6, 0.27, 60)
# plt.contour(x[1:Nx-20, :, cld(Nz, 2)], y[1:Nx-20, :, cld(Nz, 2)], YH2O[1:Nx-20, :, cld(Nz, 2)], levels=60, cmap="hot", extend="min")
# a = plt.contourf(x[1:Nx-20, :, cld(Nz, 2)], y[1:Nx-20, :, cld(Nz, 2)], T[1:Nx-20, :, cld(Nz, 2)], 60, cmap="coolwarm", extend="min")
# plt.xlabel(raw"$x/m$")
# plt.ylabel(raw"$y/m$")

# plt.colorbar(a, label=raw"$T$", location="right")
# plt.annotate(raw"$H_2O$ mass fraction", style="italic", xy=(0.015,-0.001), xytext=(0.01,-0.003), arrowprops=Dict("facecolor"=>"black"))
# # plt.tight_layout()
# plt.title("5 million grid, H2 combustion with one GPU")
# # plt.show()
# plt.savefig("view.png", dpi=600)