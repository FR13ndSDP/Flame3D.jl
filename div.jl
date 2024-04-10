function div(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z
    
    if i > Nxp || j > Nyp || k > Nzp
        return
    end

    c1::Float32 = 1/12f0
    c2::Float32 = -2/3f0

    @inbounds Jact::Float32 = J[i+NG, j+NG, k+NG] * dt

    @inbounds dV11dξ = c1*(Fv_x[i,   j+2, k+2, 1] - Fv_x[i+4, j+2, k+2, 1]) + 
                       c2*(Fv_x[i+1, j+2, k+2, 1] - Fv_x[i+3, j+2, k+2, 1])
    @inbounds dV12dξ = c1*(Fv_x[i,   j+2, k+2, 2] - Fv_x[i+4, j+2, k+2, 2]) +
                       c2*(Fv_x[i+1, j+2, k+2, 2] - Fv_x[i+3, j+2, k+2, 2])
    @inbounds dV13dξ = c1*(Fv_x[i,   j+2, k+2, 3] - Fv_x[i+4, j+2, k+2, 3]) +
                       c2*(Fv_x[i+1, j+2, k+2, 3] - Fv_x[i+3, j+2, k+2, 3])
    @inbounds dV14dξ = c1*(Fv_x[i,   j+2, k+2, 4] - Fv_x[i+4, j+2, k+2, 4]) +
                       c2*(Fv_x[i+1, j+2, k+2, 4] - Fv_x[i+3, j+2, k+2, 4])

    @inbounds dV21dη = c1*(Fv_y[i+2, j,   k+2, 1] - Fv_y[i+2, j+4, k+2, 1]) + 
                       c2*(Fv_y[i+2, j+1, k+2, 1] - Fv_y[i+2, j+3, k+2, 1])
    @inbounds dV22dη = c1*(Fv_y[i+2, j,   k+2, 2] - Fv_y[i+2, j+4, k+2, 2]) +
                       c2*(Fv_y[i+2, j+1, k+2, 2] - Fv_y[i+2, j+3, k+2, 2])
    @inbounds dV23dη = c1*(Fv_y[i+2, j,   k+2, 3] - Fv_y[i+2, j+4, k+2, 3]) +
                       c2*(Fv_y[i+2, j+1, k+2, 3] - Fv_y[i+2, j+3, k+2, 3])
    @inbounds dV24dη = c1*(Fv_y[i+2, j,   k+2, 4] - Fv_y[i+2, j+4, k+2, 4]) +
                       c2*(Fv_y[i+2, j+1, k+2, 4] - Fv_y[i+2, j+3, k+2, 4])

    @inbounds dV31dζ = c1*(Fv_z[i+2, j+2, k,   1] - Fv_z[i+2, j+2, k+4, 1]) +
                       c2*(Fv_z[i+2, j+2, k+1, 1] - Fv_z[i+2, j+2, k+3, 1])
    @inbounds dV32dζ = c1*(Fv_z[i+2, j+2, k,   2] - Fv_z[i+2, j+2, k+4, 2]) +
                       c2*(Fv_z[i+2, j+2, k+1, 2] - Fv_z[i+2, j+2, k+3, 2])
    @inbounds dV33dζ = c1*(Fv_z[i+2, j+2, k,   3] - Fv_z[i+2, j+2, k+4, 3]) +
                       c2*(Fv_z[i+2, j+2, k+1, 3] - Fv_z[i+2, j+2, k+3, 3])
    @inbounds dV34dζ = c1*(Fv_z[i+2, j+2, k,   4] - Fv_z[i+2, j+2, k+4, 4]) +
                       c2*(Fv_z[i+2, j+2, k+1, 4] - Fv_z[i+2, j+2, k+3, 4])
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