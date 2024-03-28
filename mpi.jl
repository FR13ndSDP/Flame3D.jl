# CUDA-aware MPI not available
function exchange_ghost(Q, NV, comm_cart, 
    sbuf_hx, sbuf_dx, rbuf_hx, rbuf_dx,
    sbuf_hy, sbuf_dy, rbuf_hy, rbuf_dy,
    sbuf_hz, sbuf_dz, rbuf_hz, rbuf_dz)
nthreadsx = (NG, 16, 16)
nthreadsy = (16, NG, 16)
nthreadsz = (16, 16, NG)

nblocksx = (1, cld((Nyp+2*NG), 16), cld((Nzp+2*NG), 16))
nblocksy = (cld((Nxp+2*NG), 16), 1, cld((Nzp+2*NG), 16))
nblocksz = (cld((Nxp+2*NG), 16), cld((Nyp+2*NG), 16), 1)

# x+
src, dst = MPI.Cart_shift(comm_cart, 0, 1)

if src != -1 || dst != -1
if dst != -1
@cuda threads=nthreadsx blocks=nblocksx pack_R(sbuf_dx, Q, NV)
copyto!(sbuf_hx, sbuf_dx)
end
MPI.Sendrecv!(sbuf_hx, rbuf_hx, comm_cart; dest=dst, source=src)
if src != -1
copyto!(rbuf_dx, rbuf_hx)
@cuda threads=nthreadsx blocks=nblocksx unpack_L(rbuf_dx, Q, NV)
end
end

# x-
src, dst = MPI.Cart_shift(comm_cart, 0, -1)

if src != -1 || dst != -1
if dst != -1
@cuda threads=nthreadsx blocks=nblocksx pack_L(sbuf_dx, Q, NV)
copyto!(sbuf_hx, sbuf_dx)
end
MPI.Sendrecv!(sbuf_hx, rbuf_hx, comm_cart; dest=dst, source=src)
if src != -1
copyto!(rbuf_dx, rbuf_hx)
@cuda threads=nthreadsx blocks=nblocksx unpack_R(rbuf_dx, Q, NV)
end
end

# y+
src, dst = MPI.Cart_shift(comm_cart, 1, 1)

if src != -1 || dst != -1
if dst != -1
@cuda threads=nthreadsy blocks=nblocksy pack_U(sbuf_dy, Q, NV)
copyto!(sbuf_hy, sbuf_dy)
end
MPI.Sendrecv!(sbuf_hy, rbuf_hy, comm_cart; dest=dst, source=src)
if src != -1
copyto!(rbuf_dy, rbuf_hy)
@cuda threads=nthreadsy blocks=nblocksy unpack_D(rbuf_dy, Q, NV)
end
end

# y-
src, dst = MPI.Cart_shift(comm_cart, 1, -1)

if src != -1 || dst != -1
if dst != -1
@cuda threads=nthreadsy blocks=nblocksy pack_D(sbuf_dy, Q, NV)
copyto!(sbuf_hy, sbuf_dy)
end
MPI.Sendrecv!(sbuf_hy, rbuf_hy, comm_cart; dest=dst, source=src)
if src != -1
copyto!(rbuf_dy, rbuf_hy)
@cuda threads=nthreadsy blocks=nblocksy unpack_U(rbuf_dy, Q, NV)
end
end

# z+
src, dst = MPI.Cart_shift(comm_cart, 2, 1)

if src != -1 || dst != -1
if dst != -1
@cuda threads=nthreadsz blocks=nblocksz pack_F(sbuf_dz, Q, NV)
copyto!(sbuf_hz, sbuf_dz)
end
MPI.Sendrecv!(sbuf_hz, rbuf_hz, comm_cart; dest=dst, source=src)
if src != -1
copyto!(rbuf_dz, rbuf_hz)
@cuda threads=nthreadsz blocks=nblocksz unpack_B(rbuf_dz, Q, NV)
end
end

# z-
src, dst = MPI.Cart_shift(comm_cart, 2, -1)

if src != -1 || dst != -1
if dst != -1
@cuda threads=nthreadsz blocks=nblocksz pack_B(sbuf_dz, Q, NV)
copyto!(sbuf_hz, sbuf_dz)
end
MPI.Sendrecv!(sbuf_hz, rbuf_hz, comm_cart; dest=dst, source=src)
if src != -1
copyto!(rbuf_dz, rbuf_hz)
@cuda threads=nthreadsz blocks=nblocksz unpack_F(rbuf_dz, Q, NV)
end
end
end

function pack_R(buf, Q, NV)
i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

if i > NG || j > Nyp+2*NG || k > Nzp+2*NG
return
end

for n = 1:NV
@inbounds buf[i, j, k, n] = Q[Nxp+i, j, k, n]
end
return
end

function pack_U(buf, Q, NV)
i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

if i > Nxp+2*NG || j > NG || k > Nzp+2*NG
return
end

for n = 1:NV
@inbounds buf[i, j, k, n] = Q[i, Nyp+j, k, n]
end
return
end

function pack_F(buf, Q, NV)
i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

if i > Nxp+2*NG || j > Nyp+2*NG || k > NG
return
end

for n = 1:NV
@inbounds buf[i, j, k, n] = Q[i, j, Nzp+k, n]
end
return
end

function pack_L(buf, Q, NV)
i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

if i > NG || j > Nyp+2*NG || k > Nzp+2*NG
return
end

for n = 1:NV
@inbounds buf[i, j, k, n] = Q[NG+i, j, k, n]
end
return
end

function pack_D(buf, Q, NV)
i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

if i > Nxp+2*NG || j > NG || k > Nzp+2*NG
return
end

for n = 1:NV
@inbounds buf[i, j, k, n] = Q[i, NG+j, k, n]
end
return
end

function pack_B(buf, Q, NV)
i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

if i > Nxp+2*NG || j > Nyp+2*NG || k > NG
return
end

for n = 1:NV
@inbounds buf[i, j, k, n] = Q[i, j, NG+k, n]
end
return
end

function unpack_L(buf, Q, NV)
i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

if i > NG || j > Nyp+2*NG || k > Nzp+2*NG
return
end

for n = 1:NV
@inbounds Q[i, j, k, n] = buf[i, j, k, n]
end
return
end

function unpack_D(buf, Q, NV)
i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

if i > Nxp+2*NG || j > NG || k > Nzp+2*NG
return
end

for n = 1:NV
@inbounds Q[i, j, k, n] = buf[i, j, k, n]
end
return
end

function unpack_B(buf, Q, NV)
i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

if i > Nxp+2*NG || j > Nyp+2*NG || k > NG
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

if i > NG || j > Nyp+2*NG || k > Nzp+2*NG
return
end

for n = 1:NV
@inbounds Q[i+Nxp+NG, j, k, n] = buf[i, j, k, n]
end
return
end

function unpack_U(buf, Q, NV)
i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

if i > Nxp+2*NG || j > NG || k > Nzp+2*NG
return
end

for n = 1:NV
@inbounds Q[i, j+Nyp+NG, k, n] = buf[i, j, k, n]
end
return
end

function unpack_F(buf, Q, NV)
i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

if i > Nxp+2*NG || j > Nyp+2*NG || k > NG
return
end

for n = 1:NV
@inbounds Q[i, j, k+Nzp+NG, n] = buf[i, j, k, n]
end
return
end