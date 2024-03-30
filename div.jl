function div(U, Fx, Fy, Fz, dt, J)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z
    if i > Nxp || j > Nyp || k > Nzp
        return
    end

    @inbounds Jact::Float32 = J[i+NG, j+NG, k+NG] * dt

    for n = 1:Ncons
        @inbounds U[i+NG, j+NG, k+NG, n] +=  (Fx[i, j, k, n] - Fx[i+1, j, k, n] + 
                                              Fy[i, j, k, n] - Fy[i, j+1, k, n] +
                                              Fz[i, j, k, n] - Fz[i, j, k+1, n]) * Jact
    end
    return
end
