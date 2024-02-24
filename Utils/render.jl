using GLMakie
using ReadVTK

GLMakie.activate!()

fname = "../PLT/plt0-1.vts"

vtk = VTKFile(fname)

# point data
p_data = get_point_data(vtk)

# mesh cordinate
x,y,z = get_coordinates(vtk)
Nx, Ny, Nz = size(x)

# variables
T = get_data(p_data["T"])
rho = get_data(p_data["rho"])
H2 = get_data(p_data["YH2"])
YH2 = H2./rho

T = reshape(T, (Nx, Ny, Nz))
rho = reshape(rho, (Nx, Ny, Nz))
YH2 = reshape(YH2, (Nx, Ny, Nz))


# Make a colormap, with the first value being transparent
colormap = to_colormap(:hot);
colormap[1] = RGBAf(0,0,0,0);
set_theme!(theme_black())
xx = collect(minimum(x):1e-4:maximum(x))
yy = collect(minimum(x):2e-5:maximum(y))
zz = collect(minimum(x):2e-5:maximum(z))
asp = (ceil(Int, (maximum(x)-minimum(x))/(maximum(y)-minimum(y))), 1, 1)
a = volume(xx, yy, zz, YH2, algorithm = :mip, colormap=colormap, axis=(type=Axis3, title="LES simulation of jet flow", aspect=asp))
save("view.png", a, px_per_unit=5.0)