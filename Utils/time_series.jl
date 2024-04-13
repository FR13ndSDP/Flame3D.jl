using WriteVTK, HDF5

fname = "./SAMPLE/collection-y.h5"

data = h5read(fname, "collection")

const time_step = 1.5f-8
const interval = 10
const total = 100

# mesh cordinate
# x,y,z = get_coordinates(vtk)
const NG::Int64 = h5read("metrics.h5", "NG")
const Nx::Int64 = h5read("metrics.h5", "Nx")
const Ny::Int64 = h5read("metrics.h5", "Ny")
const Nz::Int64 = h5read("metrics.h5", "Nz")

fid = h5open("metrics.h5", "r")
x = fid["x"][NG+1:Nx+NG, 130+NG, NG+1:Nz+NG] 
y = fid["y"][NG+1:Nx+NG, 130+NG, NG+1:Nz+NG] 
z = fid["z"][NG+1:Nx+NG, 130+NG, NG+1:Nz+NG]
close(fid)

# variables
p = data[:, :, 1, :]

times = range(time_step, time_step*total; step=time_step*interval)

saved_files = paraview_collection("./SAMPLE/full_simulation") do pvd
    for (n, time) ∈ enumerate(times)
        vtk_surface("./SAMPLE/timestep_$n", x, y, z) do vtk
            vtk["p"] = p[:, :, n]
            pvd[time] = vtk
        end
    end
end