using GLMakie
using ReadVTK

GLMakie.activate!()

fname = "../PLT/plt0-1.vts"

const NG::Int64 = h5read("mesh.h5", "NG")
const Nx::Int64 = h5read("mesh.h5", "Nx")
const Ny::Int64 = h5read("mesh.h5", "Ny")
const Nz::Int64 = h5read("mesh.h5", "Nz")

coords = h5read("./mesh.h5", "coords")

x = @view coords[1, :, :, :]
y = @view coords[2, :, :, :]
z = @view coords[3, :, :, :]

# variables
p = h5read(fname, "p")


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