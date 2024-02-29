using WriteVTK,HDF5

const NG::Int64 = h5read("metrics.h5", "NG")
const Nx::Int64 = h5read("metrics.h5", "Nx")
const Ny::Int64 = h5read("metrics.h5", "Ny")
const Nz::Int64 = h5read("metrics.h5", "Nz")
Nxp = Nx

fid = h5open("metrics.h5", "r")
x_h = fid["x"][:, :, :] 
y_h = fid["y"][:, :, :] 
z_h = fid["z"][:, :, :]
close(fid)


fid = h5open("./CHK/chk1500.h5", "r")
Q_h = fid["Q_h"][:, :, :, :, 1]
close(fid)


rho = convert(Array{Float32, 3}, @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:2+NG, 1])
u =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:2+NG, 2])
v =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:2+NG, 3])
w =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:2+NG, 4])
p =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:2+NG, 5])
T =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:2+NG, 6])

x_ng = convert(Array{Float32, 3}, @view x_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:2+NG])
y_ng = convert(Array{Float32, 3}, @view y_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:2+NG])
z_ng = convert(Array{Float32, 3}, @view z_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:2+NG])

vtk_grid("surface.vts", x_ng, y_ng, z_ng; compress=6) do vtk
    vtk["rho"] = rho
    vtk["u"] = u
    vtk["v"] = v
    vtk["w"] = w
    vtk["p"] = p
    vtk["T"] = T
end 