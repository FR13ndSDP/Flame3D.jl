using ReadVTK
using PyCall

fname = "../plt1300-0.vts"

plt = pyimport("matplotlib.pyplot")
plt.rcParams["font.family"] = "monospace"

vtk = VTKFile(get_example_file(fname))

# point data
p_data = get_point_data(vtk)

# mesh cordinate
x,y,z = get_coordinates(vtk)
Nx, Ny, Nz = size(x)

# variables
T = get_data(p_data["T"])
rho = get_data(p_data["rho"])
H2O = get_data(p_data["YH2O"])
YH2O = H2O./rho

T = reshape(T, (Nx, Ny, Nz))
rho = reshape(rho, (Nx, Ny, Nz))
YH2O = reshape(YH2O, (Nx, Ny, Nz))


fig = plt.figure()
ax = fig.add_subplot(211)
bx = fig.add_subplot(212)
a = ax.contourf(x[1:Nx-20, :, cld(Nz, 2)], y[1:Nx-20, :, cld(Nz, 2)], YH2O[1:Nx-20, :, cld(Nz, 2)], 120, cmap="hot")
ax.set_xlabel("x/m")
ax.set_ylabel("y/m")

b = bx.contourf(x[1:Nx-20, :, cld(Nz, 2)], y[1:Nx-20, :, cld(Nz, 2)], rho[1:Nx-20, :, cld(Nz, 2)], 120, cmap="coolwarm")
bx.set_xlabel("x/m")
bx.set_ylabel("y/m")

fig.colorbar(a, ax=ax, label=raw"$Y_{H_2O}$", location="right")
fig.colorbar(b, ax=bx, label=raw"$\rho$", location="right")
plt.tight_layout()
ax.set_title("5 million grid, H2 combustion with one GPU")
# plt.show()
plt.savefig("view.png", dpi=600)