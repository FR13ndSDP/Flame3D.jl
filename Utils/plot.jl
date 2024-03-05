using ReadVTK
using PyCall

fname = "../PLT/plt0-100.vts"

plt = pyimport("matplotlib.pyplot")
plt.rc("text", usetex= true)
plt.rc("font", family= "sans-serif")
# plt.rc("font", sans-serif = "Helvetica")
plt.rc("font", size=20)

vtk = VTKFile(fname)

# point data
p_data = get_point_data(vtk)

# mesh cordinate
x,y,z = get_coordinates(vtk)
Nx, Ny, Nz = size(x)

# variables
T = get_data_reshaped(p_data["T"])
rho = get_data_reshaped(p_data["rho"])
H2O = get_data_reshaped(p_data["YH2O"])
YH2O = H2O./rho


fig = plt.figure(figsize=(18, 6))
# lev= np.linspace(1e-6, 0.27, 60)
plt.contour(x[1:Nx-20, :, cld(Nz, 2)], y[1:Nx-20, :, cld(Nz, 2)], YH2O[1:Nx-20, :, cld(Nz, 2)], levels=60, cmap="hot", extend="min")
a = plt.contourf(x[1:Nx-20, :, cld(Nz, 2)], y[1:Nx-20, :, cld(Nz, 2)], T[1:Nx-20, :, cld(Nz, 2)], 60, cmap="coolwarm", extend="min")
plt.xlabel(raw"$x/m$")
plt.ylabel(raw"$y/m$")

plt.colorbar(a, label=raw"$T$", location="right")
plt.annotate(raw"$H_2O$ mass fraction", style="italic", xy=(0.015,-0.001), xytext=(0.01,-0.003), arrowprops=Dict("facecolor"=>"black"))
# plt.tight_layout()
plt.title("5 million grid, H2 combustion with one GPU")
# plt.show()
plt.savefig("view.png", dpi=600)