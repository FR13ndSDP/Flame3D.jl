function div(U, Fx, Fy, Fz, dt, J)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z
    if i > Nxp || j > Nyp || k > Nzp
        return
    end

    @inbounds Jact = J[i+NG, j+NG, k+NG] * dt

    for n = 1:Ncons
        @inbounds U[i+NG, j+NG, k+NG, n] +=  (Fx[i, j, k, n] - Fx[i+1, j, k, n] + 
                                              Fy[i, j, k, n] - Fy[i, j+1, k, n] +
                                              Fz[i, j, k, n] - Fz[i, j, k+1, n]) * Jact
    end
    return
end