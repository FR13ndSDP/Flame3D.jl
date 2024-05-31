"""
    div(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J)

Compute divergence of inviscous and viscous fluxes on grid point (without ghosts) 

...
# Arguments
- `U`: conservative variables
- `Fx, Fy, Fz`: inviscous fluxes
- `Fv_x, Fv_y, Fv_z`: viscous fluxes
- `dt`: time step
- `J`: det of Jacobian, can be seen as 1/Δ, where Δ is the element volume
...
"""
function div(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z
    if i > Nxp || j > Nyp || k > Nzp
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
    @inbounds U[i+NG, j+NG, k+NG, 2] += (dV11dξ + dV21dη + dV31dζ) * Jact # x-momentum
    @inbounds U[i+NG, j+NG, k+NG, 3] += (dV12dξ + dV22dη + dV32dζ) * Jact # y-momentum
    @inbounds U[i+NG, j+NG, k+NG, 4] += (dV13dξ + dV23dη + dV33dζ) * Jact # z-momentum
    @inbounds U[i+NG, j+NG, k+NG, 5] += (dV14dξ + dV24dη + dV34dζ) * Jact # Energy
    return
end

"""
    divSpecs(U, Fx, Fy, Fz, Fd_x, Fd_y, Fd_z, dt, J)

Compute divergence of fluxes for species on grid point (without ghosts) 
"""
function divSpecs(U, Fx, Fy, Fz, Fd_x, Fd_y, Fd_z, dt, J)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z
    if i > Nxp || j > Nyp || k > Nzp
        return
    end
    
    @inbounds Jact = J[i+NG, j+NG, k+NG] * dt

    for n = 1:Nspecs
        @inbounds U[i+NG, j+NG, k+NG, n] +=  (Fx[i, j, k, n] - Fx[i+1, j, k, n] + 
                                              Fy[i, j, k, n] - Fy[i, j+1, k, n] + 
                                              Fz[i, j, k, n] - Fz[i, j, k+1, n] + 
                                              Fd_x[i+1, j, k, n] - Fd_x[i, j, k, n] +
                                              Fd_y[i, j+1, k, n] - Fd_y[i, j, k, n] +
                                              Fd_z[i, j, k+1, n] - Fd_z[i, j, k, n]) * Jact
    end
    return
end