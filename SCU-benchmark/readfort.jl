using FortranFiles, WriteVTK, HDF5

const NG::Int64 = 4
const Nx::Int64 = 1000
const Ny::Int64 = 240
const Nz::Int64 = 100
const Lx::Float32 = 4
const Nx_tot::Int64 = Nx + 2*NG
const Ny_tot::Int64 = Ny + 2*NG
const Nz_tot::Int64 = Nz + 2*NG
const vis::Bool = true
const compress_level::Int64 = 3

x = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
y = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
z = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)

f = FortranFile("OCFD3d-Mesh.dat")

for n = 1:Nz
    x[NG+1:Nx+NG, NG+1:Ny+NG, n+NG] = read(f, (Float64, Nx, Ny))
end

for n = 1:Nz
    y[NG+1:Nx+NG, NG+1:Ny+NG, n+NG] = read(f, (Float64, Nx, Ny))
end

for n = 1:Nz
    z[NG+1:Nx+NG, NG+1:Ny+NG, n+NG] = read(f, (Float64, Nx, Ny))
end

# get ghost location
@inbounds for k ∈ NG+1:Nz+NG, j ∈ NG+1:Ny+NG, i ∈ 1:NG
    x[i, j, k] = 2*x[NG+1, j, k] - x[2*NG+2-i, j, k]
    y[i, j, k] = 2*y[NG+1, j, k] - y[2*NG+2-i, j, k]
    z[i, j, k] = 2*z[NG+1, j, k] - z[2*NG+2-i, j, k]
end

@inbounds for k ∈ NG+1:Nz+NG, j ∈ NG+1:Ny+NG, i ∈ Nx+NG+1:Nx_tot
    x[i, j, k] = 2*x[Nx+NG, j, k] - x[2*NG+2*Nx-i, j, k]
    y[i, j, k] = 2*y[Nx+NG, j, k] - y[2*NG+2*Nx-i, j, k]
    z[i, j, k] = 2*z[Nx+NG, j, k] - z[2*NG+2*Nx-i, j, k]
end

@inbounds for k ∈ NG+1:Nz+NG, j ∈ 1:NG, i ∈ NG+1:Nx+NG
    x[i, j, k] = 2*x[i, NG+1, k] - x[i, 2*NG+2-j, k]
    y[i, j, k] = 2*y[i, NG+1, k] - y[i, 2*NG+2-j, k]
    z[i, j, k] = z[i, 2*NG+2-j, k]
end

@inbounds for k ∈ NG+1:Nz+NG, j ∈ Ny+NG+1:Ny_tot, i ∈ NG+1:Nx+NG
    x[i, j, k] = 2*x[i, Ny+NG, k] - x[i, 2*NG+2*Ny-j, k]
    y[i, j, k] = 2*y[i, Ny+NG, k] - y[i, 2*NG+2*Ny-j, k]
    z[i, j, k] = z[i, 2*NG+2*Ny-j, k]
end

#corner ghost
@inbounds for k ∈ NG+1:Nz+NG, j ∈ Ny+NG+1:Ny_tot, i ∈ 1:NG
    x[i, j, k] = x[i, Ny+NG, k] + x[NG+1, j, k] - x[NG+1, Ny+NG, k]
    y[i, j, k] = y[i, Ny+NG, k] + y[NG+1, j, k] - y[NG+1, Ny+NG, k]
    z[i, j, k] = z[i, Ny+NG, k] + z[NG+1, j, k] - z[NG+1, Ny+NG, k]
end

@inbounds for k ∈ NG+1:Nz+NG, j ∈ 1:NG, i ∈ 1:NG
    x[i, j, k] = x[i, NG+1, k] + x[NG+1, j, k] - x[NG+1, NG+1, k]
    y[i, j, k] = y[i, NG+1, k] + y[NG+1, j, k] - y[NG+1, NG+1, k]
    z[i, j, k] = z[i, NG+1, k] + z[NG+1, j, k] - z[NG+1, NG+1, k]
end

@inbounds for k ∈ NG+1:Nz+NG, j ∈ Ny+NG+1:Ny_tot, i ∈ Nx+NG+1:Nx_tot
    x[i, j, k] = x[i, Ny+NG, k] + x[Nx+NG, j, k] - x[Nx+NG, Ny+NG, k]
    y[i, j, k] = y[i, Ny+NG, k] + y[Nx+NG, j, k] - y[Nx+NG, Ny+NG, k]
    z[i, j, k] = z[i, Ny+NG, k] + z[Nx+NG, j, k] - z[Nx+NG, Ny+NG, k]
end

@inbounds for k ∈ NG+1:Nz+NG, j ∈ 1:NG, i ∈ Nx+NG+1:Nx_tot
    x[i, j, k] = x[i, NG+1, k] + x[Nx+NG, j, k] - x[Nx+NG, NG+1, k]
    y[i, j, k] = y[i, NG+1, k] + y[Nx+NG, j, k] - y[Nx+NG, NG+1, k]
    z[i, j, k] = z[i, NG+1, k] + z[Nx+NG, j, k] - z[Nx+NG, NG+1, k]
end

@inbounds for k ∈ 1:NG, j ∈ 1:Ny_tot, i ∈ 1:Nx_tot
    x[i, j, k] = 2*x[i, j, NG+1] - x[i, j, 2*NG+2-k]
    y[i, j, k] = 2*y[i, j, NG+1] - y[i, j, 2*NG+2-k]
    z[i, j, k] = 2*z[i, j, NG+1] - z[i, j, 2*NG+2-k]
end

@inbounds for k ∈ Nz+NG+1:Nz_tot, j ∈ 1:Ny_tot, i ∈ 1:Nx_tot
    x[i, j, k] = 2*x[i, j, Nz+NG] - x[i, j, 2*NG+2*Nz-k]
    y[i, j, k] = 2*y[i, j, Nz+NG] - y[i, j, 2*NG+2*Nz-k]
    z[i, j, k] = 2*z[i, j, Nz+NG] - z[i, j, 2*NG+2*Nz-k]
end


# convert to meter
x ./= 1000.0
y ./= 1000.0
z ./= 1000.0

# compute jacobian
function CD6(f)
    fₓ = 1/60*(f[7]-f[1]) - 3/20*(f[6]-f[2]) + 3/4*(f[5]-f[3])
    return fₓ
end

function CD2_L(f)
    fₓ = 2*f[2] - 0.5*f[3] - 1.5*f[1]
    return fₓ
end

function CD2_R(f)
    fₓ = -2*f[2] + 0.5*f[1] + 1.5*f[3]
    return fₓ
end

# Jacobians
dxdξ = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
dxdη = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
dxdζ = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
dydξ = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
dydη = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
dydζ = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
dzdξ = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
dzdη = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
dzdζ = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)

dξdx = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
dηdx = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
dζdx = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
dξdy = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
dηdy = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
dζdy = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
dξdz = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
dηdz = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
dζdz = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)
J  = zeros(Float32, Nx_tot, Ny_tot, Nz_tot)

@inbounds for k ∈ 1:Nz_tot, j ∈ 1:Ny_tot, i ∈ 4:Nx_tot-3
    dxdξ[i, j, k] = CD6(x[i-3:i+3, j, k])
    dydξ[i, j, k] = CD6(y[i-3:i+3, j, k])
    dzdξ[i, j, k] = CD6(z[i-3:i+3, j, k])
end

@inbounds for k ∈ 1:Nz_tot, j ∈ 4:Ny_tot-3, i ∈ 1:Nx_tot
    dxdη[i, j, k] = CD6(x[i, j-3:j+3, k])
    dydη[i, j, k] = CD6(y[i, j-3:j+3, k])
    dzdη[i, j, k] = CD6(z[i, j-3:j+3, k])
end

@inbounds for k ∈ 4:Nz_tot-3, j ∈ 1:Ny_tot, i ∈ 1:Nx_tot
    dxdζ[i, j, k] = CD6(x[i, j, k-3:k+3])
    dydζ[i, j, k] = CD6(y[i, j, k-3:k+3])
    dzdζ[i, j, k] = CD6(z[i, j, k-3:k+3])
end

# boundary
@inbounds for k ∈ 1:Nz_tot, j ∈ 1:Ny_tot, i ∈ 1:3
    dxdξ[i, j, k] = CD2_L(x[i:i+2, j, k])
    dydξ[i, j, k] = CD2_L(y[i:i+2, j, k])
    dzdξ[i, j, k] = CD2_L(z[i:i+2, j, k])
end

@inbounds for k ∈ 1:Nz_tot, j ∈ 1:Ny_tot, i ∈ Nx_tot-2:Nx_tot
    dxdξ[i, j, k] = CD2_R(x[i-2:i, j, k])
    dydξ[i, j, k] = CD2_R(y[i-2:i, j, k])
    dzdξ[i, j, k] = CD2_R(z[i-2:i, j, k])
end

@inbounds for k ∈ 1:Nz_tot, j ∈ 1:3, i ∈ 1:Nx_tot
    dxdη[i, j, k] = CD2_L(x[i, j:j+2, k])
    dydη[i, j, k] = CD2_L(y[i, j:j+2, k])
    dzdη[i, j, k] = CD2_L(z[i, j:j+2, k])
end

@inbounds for k ∈ 1:Nz_tot, j ∈ Ny_tot-2:Ny_tot, i ∈ 1:Nx_tot
    dxdη[i, j, k] = CD2_R(x[i, j-2:j, k])
    dydη[i, j, k] = CD2_R(y[i, j-2:j, k])
    dzdη[i, j, k] = CD2_R(z[i, j-2:j, k])
end

@inbounds for k ∈ 1:3, j ∈ 1:Ny_tot, i ∈ 1:Nx_tot
    dxdζ[i, j, k] = CD2_L(x[i, j, k:k+2])
    dydζ[i, j, k] = CD2_L(y[i, j, k:k+2])
    dzdζ[i, j, k] = CD2_L(z[i, j, k:k+2])
end

@inbounds for k ∈ Nz_tot-2:Nz_tot, j ∈ 1:Ny_tot, i ∈ 1:Nx_tot
    dxdζ[i, j, k] = CD2_R(x[i, j, k-2:k])
    dydζ[i, j, k] = CD2_R(y[i, j, k-2:k])
    dzdζ[i, j, k] = CD2_R(z[i, j, k-2:k])
end

@. J = 1 / (dxdξ*(dydη*dzdζ - dydζ*dzdη) - dxdη*(dydξ*dzdζ-dydζ*dzdξ) + dxdζ*(dydξ*dzdη-dydη*dzdξ))

# actually after * J⁻
@. dξdx = dydη*dzdζ - dydζ*dzdη
@. dξdy = dxdζ*dzdη - dxdη*dzdζ
@. dξdz = dxdη*dydζ - dxdζ*dydη
@. dηdx = dydζ*dzdξ - dydξ*dzdζ
@. dηdy = dxdξ*dzdζ - dxdζ*dzdξ
@. dηdz = dxdζ*dydξ - dxdξ*dydζ
@. dζdx = dydξ*dzdη - dydη*dzdξ
@. dζdy = dxdη*dzdξ - dxdξ*dzdη
@. dζdz = dxdξ*dydη - dxdη*dydξ

h5open("metrics.h5", "w") do file
    file["NG"] = NG
    file["Nx"] = Nx
    file["Ny"] = Ny
    file["Nz"] = Nz
    file["dξdx", compress=compress_level] = dξdx
    file["dξdy", compress=compress_level] = dξdy
    file["dξdz", compress=compress_level] = dξdz
    file["dηdx", compress=compress_level] = dηdx
    file["dηdy", compress=compress_level] = dηdy
    file["dηdz", compress=compress_level] = dηdz
    file["dζdx", compress=compress_level] = dζdx
    file["dζdy", compress=compress_level] = dζdy
    file["dζdz", compress=compress_level] = dζdz
    file["J", compress=compress_level] = J
    file["x", compress=compress_level] = x
    file["y", compress=compress_level] = y
    file["z", compress=compress_level] = z
end

if vis
    vtk_grid("mesh.vts", x, y, z) do vtk
        vtk["J"] = J
        vtk["dkdx"] = dξdx
        vtk["dkdy"] = dξdy
        vtk["dkdz"] = dξdz
        vtk["dedx"] = dηdx
        vtk["dedy"] = dηdy
        vtk["dedz"] = dηdz
        vtk["dsdx"] = dζdx
        vtk["dsdy"] = dζdy
        vtk["dsdz"] = dζdz
    end

end

println("Parse mesh done!")
flush(stdout)