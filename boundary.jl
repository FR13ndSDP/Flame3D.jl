function fill_x(Q, U, rankx, ranky, inlet)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z
    
    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
        return
    end
    # inlet
    if rankx == 0 && i <= NG+1 && j >= NG+1 && j <= Nyp+NG
        T_ref = 108.1f0
        u_ref = 607.177038268082f0
        ρ_ref = 0.06894018305600735f0

        ρ = inlet[j-NG+ranky*Nyp, 1] * ρ_ref
        u = inlet[j-NG+ranky*Nyp, 2] * u_ref
        v = inlet[j-NG+ranky*Nyp, 3] * u_ref
        T = inlet[j-NG+ranky*Nyp, 4] * T_ref
        p = ρ * 287 * T

        @inbounds Q[i, j, k, 1] = ρ
        @inbounds Q[i, j, k, 2] = u
        @inbounds Q[i, j, k, 3] = v
        @inbounds Q[i, j, k, 4] = 0
        @inbounds Q[i, j, k, 6] = T
        @inbounds Q[i, j, k, 5] = p

        @inbounds U[i, j, k, 1] = ρ
        @inbounds U[i, j, k, 2] = ρ * u
        @inbounds U[i, j, k, 3] = ρ * v
        @inbounds U[i, j, k, 4] = 0
        @inbounds U[i, j, k, 5] = p/0.4f0 + 0.5f0*ρ*(u^2+v^2)
    end

    if rankx == (Nprocs[1]-1) && i > Nxp+NG
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[Nxp+NG, j, k, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[Nxp+NG, j, k, n]
        end
    end
    return
end

function fill_z(Q, U)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z
    
    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
        return
    end

    if k <= NG
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, j, k+Nzp, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, j, k+Nzp, n]
        end
    elseif k > Nzp+NG
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, j, k-Nzp, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, j, k-Nzp, n]
        end
    end
    return
end

function fill_y(Q, U, rankx, ranky)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z
    
    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
        return
    end


    if ranky == 0 && j == NG+1

        if rankx == 0 && i >= 50 && i <= 70
            v_turb::Float32 = sin((i-50)/20*2π)*cos(k/100*4π)*600f0*0.1f0
        else
            v_turb = 0.f0
        end

        pw = 1.5f0*Q[i, j+1, k, 5] - 0.5f0*Q[i, j+2, k, 5]
        Tw = 307.f0
        ρw = pw/(287 * Tw)
        @inbounds Q[i, j, k, 5] = pw
        @inbounds Q[i, j, k, 1] = ρw
        @inbounds Q[i, j, k, 2] = 0
        @inbounds Q[i, j, k, 3] = v_turb
        @inbounds Q[i, j, k, 4] = 0
        @inbounds Q[i, j, k, 6] = Tw

        @inbounds U[i, j, k, 1] = ρw
        @inbounds U[i, j, k, 2] = 0
        @inbounds U[i, j, k, 3] = ρw * v_turb
        @inbounds U[i, j, k, 4] = 0
        @inbounds U[i, j, k, 5] = pw/0.4f0 + 0.5*ρw*v_turb^2

        for l = NG:-1:1
            u = -Q[i, 2*NG+2-l, k, 2]
            v = -Q[i, 2*NG+2-l, k, 3] + 2*v_turb
            w = -Q[i, 2*NG+2-l, k, 4]
            T = 2*Tw - Q[i, 2*NG+2-l, k, 6]
            p = 1.5f0*Q[i, l+1, k, 5] - 0.5f0*Q[i, l+2, k, 5]
            ρ = p/(287*T)

            @inbounds Q[i, l, k, 5] = p
            @inbounds Q[i, l, k, 1] = ρ
            @inbounds Q[i, l, k, 2] = u
            @inbounds Q[i, l, k, 3] = v
            @inbounds Q[i, l, k, 4] = w
            @inbounds Q[i, l, k, 6] = T

            @inbounds U[i, l, k, 1] = ρ
            @inbounds U[i, l, k, 2] = ρ * u
            @inbounds U[i, l, k, 3] = ρ * v
            @inbounds U[i, l, k, 4] = ρ * w
            @inbounds U[i, l, k, 5] = p/0.4f0 + 0.5f0 * ρ * (u^2+v^2+w^2)
        end
    elseif ranky == (Nprocs[2]-1) && j > Nyp+NG
        for n = 1:Nprim
            @inbounds Q[i, j, k, n] = Q[i, Nyp+NG, k, n]
        end
        for n = 1:Ncons
            @inbounds U[i, j, k, n] = U[i, Nyp+NG, k, n]
        end
    end
    return
end

# special treatment on wall
# fill Q and U
function fillGhost(Q, U, rankx, ranky, inlet)
    @roc groupsize=nthreads gridsize=ngroups fill_x(Q, U, rankx, ranky, inlet)
    @roc groupsize=nthreads gridsize=ngroups fill_y(Q, U, rankx, ranky)
    # @roc groupsize=nthreads gridsize=ngroups fill_z(Q, U)
end


function init(Q, inlet, ranky)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > Nxp+2*NG || j > Nyp+NG || j < NG+1 || k > Nzp+2*NG
        return
    end

    T_ref = 108.1f0
    u_ref = 607.177038268082f0
    ρ_ref = 0.06894018305600735f0

    ρ = inlet[j-NG+ranky*Nyp, 1] * ρ_ref
    u = inlet[j-NG+ranky*Nyp, 2] * u_ref
    v = inlet[j-NG+ranky*Nyp, 3] * u_ref
    T = inlet[j-NG+ranky*Nyp, 4] * T_ref
    p = ρ * 287 * T

    @inbounds Q[i, j, k, 1] = ρ
    @inbounds Q[i, j, k, 2] = u
    @inbounds Q[i, j, k, 3] = v
    @inbounds Q[i, j, k, 4] = 0
    @inbounds Q[i, j, k, 6] = T
    @inbounds Q[i, j, k, 5] = p
    return
end

#initialization on GPU
function initialize(Q, inlet, ranky)
    @roc groupsize=nthreads gridsize=ngroups init(Q, inlet, ranky)
end