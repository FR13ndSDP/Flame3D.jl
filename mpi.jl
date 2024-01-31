# CUDA-aware MPI not available
function exchange_ghost(Q, NV, rank, comm, sbuf_h, sbuf_d, rbuf_h, rbuf_d)
    nthreads = (NG, 8, 8)
    nblock = (1, cld((Ny+2*NG), 8), cld((Nz+2*NG), 8))

    # x+
    src = (rank - 1 == -1 ? MPI.PROC_NULL : (rank - 1)) 
    dst = (rank + 1 == Nprocs ? MPI.PROC_NULL : (rank + 1))

    if src != MPI.PROC_NULL || dst != MPI.PROC_NULL
        if dst != MPI.PROC_NULL
            @cuda threads=nthreads blocks=nblock pack_R(sbuf_d, Q, NV)
            copyto!(sbuf_h, sbuf_d)
        end
        MPI.Sendrecv!(sbuf_h, rbuf_h, comm; dest=dst, source=src)
        if src != MPI.PROC_NULL
            copyto!(rbuf_d, rbuf_h)
            @cuda threads=nthreads blocks=nblock unpack_L(rbuf_d, Q, NV)
        end
    end

    # x-
    src = (rank + 1 == Nprocs ? MPI.PROC_NULL : (rank + 1)) 
    dst = (rank - 1 == -1 ? MPI.PROC_NULL : (rank - 1))

    if src != MPI.PROC_NULL || dst != MPI.PROC_NULL
        if dst != MPI.PROC_NULL
            @cuda threads=nthreads blocks=nblock pack_L(sbuf_d, Q, NV)
            copyto!(sbuf_h, sbuf_d)
        end
        MPI.Sendrecv!(sbuf_h, rbuf_h, comm; dest=dst, source=src)
        if src != MPI.PROC_NULL
            copyto!(rbuf_d, rbuf_h)
            @cuda threads=nthreads blocks=nblock unpack_R(rbuf_d, Q, NV)
        end
    end
end

function pack_R(buf, Q, NV)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    for n = 1:NV
        @inbounds buf[i, j, k, n] = Q[Nxp+i, j, k, n]
    end
    return
end

function pack_L(buf, Q, NV)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    for n = 1:NV
        @inbounds buf[i, j, k, n] = Q[NG+i, j, k, n]
    end
    return
end

function unpack_L(buf, Q, NV)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    for n = 1:NV
        @inbounds Q[i, j, k, n] = buf[i, j, k, n]
    end
    return
end

function unpack_R(buf, Q, NV)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    for n = 1:NV
        @inbounds Q[i+Nxp+NG, j, k, n] = buf[i, j, k, n]
    end
    return
end