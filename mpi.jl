# CUDA-aware MPI not available
function exchange_ghost(Q, NV, rank, comm)
    sendbuf_h = zeros(Float64, NG, Ny+2*NG, Nz+2*NG, NV)
    recvbuf_h = zeros(Float64, NG, Ny+2*NG, Nz+2*NG, NV)
    sendbuf_d = CuArray(sendbuf_h)
    recvbuf_d = CuArray(recvbuf_h)

    nthreads = (NG, 8, 8)
    nblock = (1, cld((Ny+2*NG), 8), cld((Nz+2*NG), 8))

    # x+
    src = (rank - 1 == -1 ? MPI.PROC_NULL : (rank - 1)) 
    dst = (rank + 1 == Nprocs ? MPI.PROC_NULL : (rank + 1))

    if src != MPI.PROC_NULL || dst != MPI.PROC_NULL
        if dst != MPI.PROC_NULL
            @cuda threads=nthreads blocks=nblock pack_R(sendbuf_d, Q, NV)
            copyto!(sendbuf_h, sendbuf_d)
        end
        MPI.Sendrecv!(sendbuf_h, recvbuf_h, comm; dest=dst, source=src)
        if src != MPI.PROC_NULL
            copyto!(recvbuf_d, recvbuf_h)
            @cuda threads=nthreads blocks=nblock unpack_L(recvbuf_d, Q, NV)
        end
    end

    # x-
    src = (rank + 1 == Nprocs ? MPI.PROC_NULL : (rank + 1)) 
    dst = (rank - 1 == -1 ? MPI.PROC_NULL : (rank - 1))

    if src != MPI.PROC_NULL || dst != MPI.PROC_NULL
        if dst != MPI.PROC_NULL
            @cuda threads=nthreads blocks=nblock pack_L(sendbuf_d, Q, NV)
            copyto!(sendbuf_h, sendbuf_d)
        end
        MPI.Sendrecv!(sendbuf_h, recvbuf_h, comm; dest=dst, source=src)
        if src != MPI.PROC_NULL
            copyto!(recvbuf_d, recvbuf_h)
            @cuda threads=nthreads blocks=nblock unpack_R(recvbuf_d, Q, NV)
        end
    end
end

function pack_R(buf, Q, NV)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    for n = 1:NV
        @inbounds buf[i, j, k, n] = Q[Nxp+i, j, k, n]
    end
    return
end

function pack_L(buf, Q, NV)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    for n = 1:NV
        @inbounds buf[i, j, k, n] = Q[NG+i, j, k, n]
    end
    return
end

function unpack_L(buf, Q, NV)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    for n = 1:NV
        @inbounds Q[i, j, k, n] = buf[i, j, k, n]
    end
    return
end

function unpack_R(buf, Q, NV)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    for n = 1:NV
        @inbounds Q[i+Nxp+NG, j, k, n] = buf[i, j, k, n]
    end
    return
end