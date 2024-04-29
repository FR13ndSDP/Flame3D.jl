using ReadVTK, WriteVTK, HDF5

fname = "./PLT/plt-200.h5"

# mesh cordinate
# x,y,z = get_coordinates(vtk)
const NG::Int64 = h5read("mesh.h5", "NG")
const Nx::Int64 = h5read("mesh.h5", "Nx")
const Ny::Int64 = h5read("mesh.h5", "Ny")
const Nz::Int64 = h5read("mesh.h5", "Nz")

coords = h5read("./mesh.h5", "coords")

x = @view coords[1, :, :, 1]
y = @view coords[2, :, :, 1]
z = @view coords[3, :, :, 1]

# variables
p = h5read(fname, "p")

p_slice = @view p[:, :, 1]

vtk_surface("slice", x, y, z) do vtk
    vtk["p"] = p_slice
end