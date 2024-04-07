using ReadVTK, WriteVTK, HDF5

fname = "./plt-500.pvts"

vtk = PVTKFile(fname)

# point data
p_data = get_point_data(vtk)

# mesh cordinate
# x,y,z = get_coordinates(vtk)
const NG::Int64 = h5read("metrics.h5", "NG")
const Nx::Int64 = h5read("metrics.h5", "Nx")
const Ny::Int64 = h5read("metrics.h5", "Ny")
const Nz::Int64 = h5read("metrics.h5", "Nz")

fid = h5open("metrics.h5", "r")
x = fid["x"][NG+1:Nx+NG, NG+1:Ny+NG, NG+1] 
y = fid["y"][NG+1:Nx+NG, NG+1:Ny+NG, NG+1] 
z = fid["z"][NG+1:Nx+NG, NG+1:Ny+NG, NG+1]
close(fid)

# variables
p = get_data_reshaped(p_data["p"])

p_slice = @view p[:, :, 1]

vtk_surface("slice", x, y, z) do vtk
    vtk["p"] = p_slice
end