function div(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    if i > Nxp || j > Ny || k > Nz
        return
    end

    @inbounds Jact = J[i+NG, j+NG, k+NG] * dt
    @inbounds dV11dξ = Fv_x[i+1, j, k, 1] - Fv_x[i, j, k, 1]
    @inbounds dV12dξ = Fv_x[i+1, j, k, 2] - Fv_x[i, j, k, 2]
    @inbounds dV13dξ = Fv_x[i+1, j, k, 3] - Fv_x[i, j, k, 3]
    @inbounds dV14dξ = Fv_x[i+1, j, k, 4] - Fv_x[i, j, k, 4]

    @inbounds dV21dη = Fv_y[i, j+1, k, 1] - Fv_y[i, j, k, 1]
    @inbounds dV22dη = Fv_y[i, j+1, k, 2] - Fv_y[i, j, k, 2]
    @inbounds dV23dη = Fv_y[i, j+1, k, 3] - Fv_y[i, j, k, 3]
    @inbounds dV24dη = Fv_y[i, j+1, k, 4] - Fv_y[i, j, k, 4]

    @inbounds dV31dζ = Fv_z[i, j, k+1, 1] - Fv_z[i, j, k, 1]
    @inbounds dV32dζ = Fv_z[i, j, k+1, 2] - Fv_z[i, j, k, 2]
    @inbounds dV33dζ = Fv_z[i, j, k+1, 3] - Fv_z[i, j, k, 3]
    @inbounds dV34dζ = Fv_z[i, j, k+1, 4] - Fv_z[i, j, k, 4]

    for n = 1:Ncons
        @inbounds U[i+NG, j+NG, k+NG, n] +=  (Fx[i, j, k, n] - Fx[i+1, j, k, n] + 
                                              Fy[i, j, k, n] - Fy[i, j+1, k, n] +
                                              Fz[i, j, k, n] - Fz[i, j, k+1, n]) * Jact
    end
    @inbounds U[i+NG, j+NG, k+NG, 2] += (dV11dξ + dV21dη + dV31dζ) * Jact
    @inbounds U[i+NG, j+NG, k+NG, 3] += (dV12dξ + dV22dη + dV32dζ) * Jact
    @inbounds U[i+NG, j+NG, k+NG, 4] += (dV13dξ + dV23dη + dV33dζ) * Jact
    @inbounds U[i+NG, j+NG, k+NG, 5] += (dV14dξ + dV24dη + dV34dζ) * Jact
    return
end
