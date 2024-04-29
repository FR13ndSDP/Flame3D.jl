using WriteVTK, HDF5

fname = "./SAMPLE/collection-z.h5"
mesh = "./mesh.h5"

data = h5read(fname, "collection")

const time_step = 1.5f-8
const sample_step = 100
const sample_total = 100

# mesh cordinate
# x,y,z = get_coordinates(vtk)
const NG::Int64 = h5read(mesh, "NG")
const Nx::Int64 = h5read(mesh, "Nx")
const Ny::Int64 = h5read(mesh, "Ny")
const Nz::Int64 = h5read(mesh, "Nz")

coords = h5read(mesh, "coords")

x = @view coords[1, :, :, 50]
y = @view coords[2, :, :, 50]
z = @view coords[3, :, :, 50]

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