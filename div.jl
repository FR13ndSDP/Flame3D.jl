#Range: 2+NG -> N+NG-1
function div(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J, consts)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z
    if i > Nx+NG-1 || i < 2+NG || j > Ny+NG-1 || j < 2+NG || k > Nz+NG-1 || k < 2+NG 
        return
    end

    c1::Float64 = consts.CD6[1]
    c2::Float64 = consts.CD6[2]
    c3::Float64 = consts.CD6[3]

    @inbounds Jac = J[i, j, k]
    @inbounds dV11dξ = c1*(Fv_x[i-1-NG, j+2-NG, k+2-NG, 1] - Fv_x[i+5-NG, j+2-NG, k+2-NG, 1]) + 
                       c2*(Fv_x[i-NG,   j+2-NG, k+2-NG, 1] - Fv_x[i+4-NG, j+2-NG, k+2-NG, 1]) + 
                       c3*(Fv_x[i+1-NG, j+2-NG, k+2-NG, 1] - Fv_x[i+3-NG, j+2-NG, k+2-NG, 1])
    @inbounds dV12dξ = c1*(Fv_x[i-1-NG, j+2-NG, k+2-NG, 2] - Fv_x[i+5-NG, j+2-NG, k+2-NG, 2]) +
                       c2*(Fv_x[i-NG,   j+2-NG, k+2-NG, 2] - Fv_x[i+4-NG, j+2-NG, k+2-NG, 2]) +
                       c3*(Fv_x[i+1-NG, j+2-NG, k+2-NG, 2] - Fv_x[i+3-NG, j+2-NG, k+2-NG, 2])
    @inbounds dV13dξ = c1*(Fv_x[i-1-NG, j+2-NG, k+2-NG, 3] - Fv_x[i+5-NG, j+2-NG, k+2-NG, 3]) +
                       c2*(Fv_x[i-NG,   j+2-NG, k+2-NG, 3] - Fv_x[i+4-NG, j+2-NG, k+2-NG, 3]) +
                       c3*(Fv_x[i+1-NG, j+2-NG, k+2-NG, 3] - Fv_x[i+3-NG, j+2-NG, k+2-NG, 3])
    @inbounds dV14dξ = c1*(Fv_x[i-1-NG, j+2-NG, k+2-NG, 4] - Fv_x[i+5-NG, j+2-NG, k+2-NG, 4]) +
                       c2*(Fv_x[i-NG,   j+2-NG, k+2-NG, 4] - Fv_x[i+4-NG, j+2-NG, k+2-NG, 4]) +
                       c3*(Fv_x[i+1-NG, j+2-NG, k+2-NG, 4] - Fv_x[i+3-NG, j+2-NG, k+2-NG, 4])

    @inbounds dV21dη = c1*(Fv_y[i+2-NG, j-1-NG, k+2-NG, 1] - Fv_y[i+2-NG, j+5-NG, k+2-NG, 1]) + 
                       c2*(Fv_y[i+2-NG, j-NG,   k+2-NG, 1] - Fv_y[i+2-NG, j+4-NG, k+2-NG, 1]) + 
                       c3*(Fv_y[i+2-NG, j+1-NG, k+2-NG, 1] - Fv_y[i+2-NG, j+3-NG, k+2-NG, 1])
    @inbounds dV22dη = c1*(Fv_y[i+2-NG, j-1-NG, k+2-NG, 2] - Fv_y[i+2-NG, j+5-NG, k+2-NG, 2]) +
                       c2*(Fv_y[i+2-NG, j-NG,   k+2-NG, 2] - Fv_y[i+2-NG, j+4-NG, k+2-NG, 2]) +
                       c3*(Fv_y[i+2-NG, j+1-NG, k+2-NG, 2] - Fv_y[i+2-NG, j+3-NG, k+2-NG, 2])
    @inbounds dV23dη = c1*(Fv_y[i+2-NG, j-1-NG, k+2-NG, 3] - Fv_y[i+2-NG, j+5-NG, k+2-NG, 3]) +
                       c2*(Fv_y[i+2-NG, j-NG,   k+2-NG, 3] - Fv_y[i+2-NG, j+4-NG, k+2-NG, 3]) +
                       c3*(Fv_y[i+2-NG, j+1-NG, k+2-NG, 3] - Fv_y[i+2-NG, j+3-NG, k+2-NG, 3])
    @inbounds dV24dη = c1*(Fv_y[i+2-NG, j-1-NG, k+2-NG, 4] - Fv_y[i+2-NG, j+5-NG, k+2-NG, 4]) +
                       c2*(Fv_y[i+2-NG, j-NG,   k+2-NG, 4] - Fv_y[i+2-NG, j+4-NG, k+2-NG, 4]) +
                       c3*(Fv_y[i+2-NG, j+1-NG, k+2-NG, 4] - Fv_y[i+2-NG, j+3-NG, k+2-NG, 4])

    @inbounds dV31dζ = c1*(Fv_z[i+2-NG, j+2-NG, k-1-NG, 1] - Fv_z[i+2-NG, j+2-NG, k+5-NG, 1]) +
                       c2*(Fv_z[i+2-NG, j+2-NG, k-NG,   1] - Fv_z[i+2-NG, j+2-NG, k+4-NG, 1]) +
                       c3*(Fv_z[i+2-NG, j+2-NG, k+1-NG, 1] - Fv_z[i+2-NG, j+2-NG, k+3-NG, 1])
    @inbounds dV32dζ = c1*(Fv_z[i+2-NG, j+2-NG, k-1-NG, 2] - Fv_z[i+2-NG, j+2-NG, k+5-NG, 2]) +
                       c2*(Fv_z[i+2-NG, j+2-NG, k-NG,   2] - Fv_z[i+2-NG, j+2-NG, k+4-NG, 2]) +
                       c3*(Fv_z[i+2-NG, j+2-NG, k+1-NG, 2] - Fv_z[i+2-NG, j+2-NG, k+3-NG, 2])
    @inbounds dV33dζ = c1*(Fv_z[i+2-NG, j+2-NG, k-1-NG, 3] - Fv_z[i+2-NG, j+2-NG, k+5-NG, 3]) +
                       c2*(Fv_z[i+2-NG, j+2-NG, k-NG,   3] - Fv_z[i+2-NG, j+2-NG, k+4-NG, 3]) +
                       c3*(Fv_z[i+2-NG, j+2-NG, k+1-NG, 3] - Fv_z[i+2-NG, j+2-NG, k+3-NG, 3])
    @inbounds dV34dζ = c1*(Fv_z[i+2-NG, j+2-NG, k-1-NG, 4] - Fv_z[i+2-NG, j+2-NG, k+5-NG, 4]) +
                       c2*(Fv_z[i+2-NG, j+2-NG, k-NG,   4] - Fv_z[i+2-NG, j+2-NG, k+4-NG, 4]) +
                       c3*(Fv_z[i+2-NG, j+2-NG, k+1-NG, 4] - Fv_z[i+2-NG, j+2-NG, k+3-NG, 4])
    for n = 1:Ncons
        @inbounds U[i, j, k, n] +=  (Fx[i-1-NG, j-1-NG, k-1-NG, n] - Fx[i-NG, j-1-NG, k-1-NG, n] + 
                                     Fy[i-1-NG, j-1-NG, k-1-NG, n] - Fy[i-1-NG, j-NG, k-1-NG, n] +
                                     Fz[i-1-NG, j-1-NG, k-1-NG, n] - Fz[i-1-NG, j-1-NG, k-NG, n]) * dt * Jac
    end
    @inbounds U[i, j, k, 2] += (dV11dξ + dV21dη + dV31dζ) * dt * Jac
    @inbounds U[i, j, k, 3] += (dV12dξ + dV22dη + dV32dζ) * dt * Jac
    @inbounds U[i, j, k, 4] += (dV13dξ + dV23dη + dV33dζ) * dt * Jac
    @inbounds U[i, j, k, 5] += (dV14dξ + dV24dη + dV34dζ) * dt * Jac
    return
end

#Range: 2+NG -> N+NG-1
function divSpecs(U, Fx, Fy, Fz, Fd_x, Fd_y, Fd_z, dt, J, consts)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z
    if i > Nx+NG-1 || i < 2+NG || j > Ny+NG-1 || j < 2+NG || k > Nz+NG-1 || k < 2+NG
        return
    end

    @inbounds Jac = J[i, j, k]
    c1::Float64 = consts.CD4[1]
    c2::Float64 = consts.CD4[2]

    for n = 1:Nspecs
        @inbounds dVdξ = c1 * (Fd_x[i-NG,   j+2-NG, k+2-NG, n] - Fd_x[i+4-NG, j+2-NG, k+2-NG, n]) +
                         c2 * (Fd_x[i+1-NG, j+2-NG, k+2-NG, n] - Fd_x[i+3-NG, j+2-NG, k+2-NG, n])
        @inbounds dVdη = c1 * (Fd_y[i+2-NG, j-NG,   k+2-NG, n] - Fd_y[i+2-NG, j+4-NG, k+2-NG, n]) +
                         c2 * (Fd_y[i+2-NG, j+1-NG, k+2-NG, n] - Fd_y[i+2-NG, j+3-NG, k+2-NG, n])
        @inbounds dVdζ = c1 * (Fd_z[i+2-NG, j+2-NG, k-NG,   n] - Fd_z[i+2-NG, j+2-NG, k+4-NG, n]) +
                         c2 * (Fd_z[i+2-NG, j+2-NG, k+1-NG, n] - Fd_z[i+2-NG, j+2-NG, k+3-NG, n])

        @inbounds U[i, j, k, n] +=  (Fx[i-1-NG, j-1-NG, k-1-NG, n] - Fx[i-NG, j-1-NG, k-1-NG, n] + 
                                     Fy[i-1-NG, j-1-NG, k-1-NG, n] - Fy[i-1-NG, j-NG, k-1-NG, n] + 
                                     Fz[i-1-NG, j-1-NG, k-1-NG, n] - Fz[i-1-NG, j-1-NG, k-NG, n] + 
                                     dVdξ + dVdη + dVdζ) * dt * Jac
    end
    return
end
