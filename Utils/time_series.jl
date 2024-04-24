using WriteVTK, HDF5

fname = "./SAMPLE/collection-z.h5"

data = h5read(fname, "collection")

const time_step = 1.5f-8
const sample_step = 100
const sample_total = 100

# mesh cordinate
# x,y,z = get_coordinates(vtk)
const NG::Int64 = h5read("metrics.h5", "NG")
const Nx::Int64 = h5read("metrics.h5", "Nx")
const Ny::Int64 = h5read("metrics.h5", "Ny")
const Nz::Int64 = h5read("metrics.h5", "Nz")

fid = h5open("metrics.h5", "r")
x = fid["x"][1+NG:Nx+NG, 1+NG:Ny+NG, 50+NG] 
y = fid["y"][1+NG:Nx+NG, 1+NG:Ny+NG, 50+NG] 
z = fid["z"][1+NG:Nx+NG, 1+NG:Ny+NG, 50+NG]
close(fid)

times = range(time_step, time_step*sample_step*sample_total; step=time_step*sample_step)

saved_files = paraview_collection("./SAMPLE/full_simulation") do pvd
    for (n, time) ∈ enumerate(times)
        vtk_surface("./SAMPLE/timestep_$n", x, y, z) do vtk
            vtk["ρ"] =  data[:, :, 1, n]
            vtk["u"] =  data[:, :, 2, n]
            vtk["v"] =  data[:, :, 3, n]
            vtk["w"] =  data[:, :, 4, n]
            vtk["p"] =  data[:, :, 5, n]
            vtk["T"] =  data[:, :, 6, n]
            pvd[time] = vtk
        end
    end
end