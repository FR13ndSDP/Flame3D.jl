using WriteVTK
using CUDA
using JLD2, JSON

CUDA.allowscalar(false)
include("split.jl")
include("schemes.jl")
include("viscous.jl")
include("boundary.jl")
include("utils.jl")
include("div.jl")

function correction(U, ρi, NG, Nx, Ny, Nz)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z
    if i > Nx+2*NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    @inbounds ρ = U[i, j, k, 1]
    ∑ρ = 0
    for n = 1:Nspecs
        @inbounds ∑ρ += ρi[i, j, k, n]
    end
    for n = 1:Nspecs
        @inbounds ρi[i, j, k, n] *= ρ/∑ρ
    end
end

function flowAdvance(U, ρi, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt)
    nthreads = (8, 8, 4)
    nblock = (cld((Nx+2*NG), 8), 
              cld((Ny+2*NG), 8),
              cld((Nz+2*NG), 4))

    # local constants
    gamma::Float64 = 1.4
    Ru::Float64 = 8.31446
    eos_m::Float64 = 28.97e-3
    Rg::Float64 = Ru/eos_m
    Pr::Float64 = 0.72
    Cv::Float64 = Rg/(gamma-1)
    Cp::Float64 = gamma * Cv
    C_s::Float64 = 1.458e-6
    T_s::Float64 = 110.4

    @cuda threads=nthreads blocks=nblock c2Prim(U, Q, Nx, Ny, Nz, NG, 1.4, 287)

    @cuda threads=nthreads blocks=nblock fluxSplit(Q, U, Fp, Fm, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz)
    @cuda maxregs=32 threads=nthreads blocks=nblock WENO_x(Fx, Fp, Fm, NG, Nx, Ny, Nz, Ncons)

    @cuda threads=nthreads blocks=nblock fluxSplit(Q, U, Fp, Fm, Nx, Ny, Nz, NG, dηdx, dηdy, dηdz)
    @cuda maxregs=32 threads=nthreads blocks=nblock WENO_y(Fy, Fp, Fm, NG, Nx, Ny, Nz, Ncons)

    @cuda threads=nthreads blocks=nblock fluxSplit(Q, U, Fp, Fm, Nx, Ny, Nz, NG, dζdx, dζdy, dζdz)
    @cuda maxregs=32 threads=nthreads blocks=nblock WENO_z(Fz, Fp, Fm, NG, Nx, Ny, Nz, Ncons)

    @cuda threads=nthreads blocks=nblock viscousFlux(Fv_x, Fv_y, Fv_z, Q, NG, Nx, Ny, Nz, Pr, Cp, C_s, T_s, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J)

    @cuda threads=nthreads blocks=nblock div(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, NG, Nx, Ny, Nz, J)

end


function specAdvance(U, ρi, Q, Fp_i, Fm_i, Fx_i, Fy_i, Fz_i, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt)
    nthreads = (8, 8, 4)
    nblock = (cld((Nx+2*NG), 8), 
              cld((Ny+2*NG), 8),
              cld((Nz+2*NG), 4))

    # @cuda threads=nthreads blocks=nblock c2Prim(U, Q, Nx, Ny, Nz, NG, 1.4, 287)

    # @cuda threads=nthreads blocks=nblock split(ρi, Q, U, Fp_i, Fm_i, dξdx, dξdy, dξdz, Nx, Ny, Nz, NG)
    # @cuda maxregs=32 threads=nthreads blocks=nblock NND_x(Fx_i, Fp_i, Fm_i, NG, Nx, Ny, Nz, Nspecs)

    # @cuda threads=nthreads blocks=nblock split(ρi, Q, U, Fp_i, Fm_i, dηdx, dηdy, dηdz, Nx, Ny, Nz, NG)
    # @cuda maxregs=32 threads=nthreads blocks=nblock NND_y(Fy_i, Fp_i, Fm_i, NG, Nx, Ny, Nz, Nspecs)

    # @cuda threads=nthreads blocks=nblock split(ρi, Q, U, Fp_i, Fm_i, dζdx, dζdy, dζdz, Nx, Ny, Nz, NG)
    # @cuda maxregs=32 threads=nthreads blocks=nblock NND_z(Fy_i, Fp_i, Fm_i, NG, Nx, Ny, Nz, Nspecs)

    # @cuda threads=nthreads blocks=nblock divSpecs(ρi, Fx_i, Fy_i, Fz_i, dt, NG, Nx, Ny, Nz, J)
end

function time_step(U, ρi, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, Nx, Ny, Nz, NG, dt)
    Nx_tot = Nx+2*NG
    Ny_tot = Ny+2*NG
    Nz_tot = Nz+2*NG

    Un = copy(U)
    # ρn = copy(ρi)
    Q =    CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nprim)
    Fp =   CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fm =   CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fx =   CUDA.zeros(Float64, Nx-1, Ny-2, Nz-2, Ncons)
    Fy =   CUDA.zeros(Float64, Nx-2, Ny-1, Nz-2, Ncons)
    Fz =   CUDA.zeros(Float64, Nx-2, Ny-2, Nz-1, Ncons)
    Fv_x = CUDA.zeros(Float64, Nx_tot-4, Ny_tot-4, Nz_tot-4, 4)
    Fv_y = CUDA.zeros(Float64, Nx_tot-4, Ny_tot-4, Nz_tot-4, 4)
    Fv_z = CUDA.zeros(Float64, Nx_tot-4, Ny_tot-4, Nz_tot-4, 4)
    # Fp_i = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
    # Fm_i = CUDA.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
    # Fx_i = CUDA.zeros(Float64, Nx-1, Ny-2, Nz-2, Nspecs)
    # Fy_i = CUDA.zeros(Float64, Nx-2, Ny-1, Nz-2, Nspecs)
    # Fz_i = CUDA.zeros(Float64, Nx-2, Ny-2, Nz-1, Nspecs)

    nthreads = (8, 8, 4)
    nblock = (cld((Nx+2*NG), 8), 
              cld((Ny+2*NG), 8),
              cld((Nz+2*NG), 4))

    for tt ∈ 1:ceil(Int, Time/dt)
        if tt % 100 == 0
            printstyled("Step: ", color=:cyan)
            print("$tt")
            printstyled("\tTime: ", color=:blue)
            println("$(tt*dt)")
            if any(isnan, U)
                printstyled("Oops, NaN detected\n", color=:red)
                return
            end
        end

        # RK3-1
        @cuda threads=nthreads blocks=nblock copyOld(Un, U, Nx, Ny, Nz, NG, Ncons)
        # @cuda threads=nthreads blocks=nblock copyOld(ρn, ρi, Nx, Ny, Nz, NG, Nspecs)

        # specAdvance(U, ρi, Q, Fp_i, Fm_i, Fx_i, Fy_i, Fz_i, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt)

        flowAdvance(U, ρi, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt)

        # @cuda threads=nthreads blocks=nblock fillSpec(ρi, U, NG, Nx, Ny, Nz)
        @cuda threads=nthreads blocks=nblock fillGhost(U, NG, Nx, Ny, Nz)

        # RK3-2
        # specAdvance(U, ρi, Q, Fp_i, Fm_i, Fx_i, Fy_i, Fz_i, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt)

        flowAdvance(U, ρi, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt)

        @cuda threads=nthreads blocks=nblock linComb(U, Un, Nx, Ny, Nz, NG, Ncons, 0.25, 0.75)
        # @cuda threads=nthreads blocks=nblock linComb(ρi, ρn, Nx, Ny, Nz, NG, Nspecs, 0.25, 0.75)
        # @cuda threads=nthreads blocks=nblock fillSpec(ρi, U, NG, Nx, Ny, Nz)
        @cuda threads=nthreads blocks=nblock fillGhost(U, NG, Nx, Ny, Nz)

        # RK3-3
        # specAdvance(U, ρi, Q, Fp_i, Fm_i, Fx_i, Fy_i, Fz_i, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt)

        flowAdvance(U, ρi, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, Nx, Ny, Nz, NG, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt)

        @cuda threads=nthreads blocks=nblock linComb(U, Un, Nx, Ny, Nz, NG, Ncons, 2/3, 1/3)
        # @cuda threads=nthreads blocks=nblock linComb(ρi, ρn, Nx, Ny, Nz, NG, Nspecs, 2/3, 1/3)
        # @cuda threads=nthreads blocks=nblock fillSpec(ρi, U, NG, Nx, Ny, Nz)
        @cuda threads=nthreads blocks=nblock fillGhost(U, NG, Nx, Ny, Nz)
        # @cuda threads=nthreads blocks=nblock correction(U, ρi, NG, Nx, Ny, Nz)
    end
    return
end
