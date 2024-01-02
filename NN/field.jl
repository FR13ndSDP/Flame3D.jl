using Plots, CUDA
using JSON, PyCall
using Lux, LuxCUDA, JLD2

dev_cpu = cpu_device()
dev_gpu = gpu_device()


@load "luxmodel.jld2" model ps st

ps = ps |> dev_gpu
st = st |> dev_gpu

const Nx::Int64 = 5
const Ny::Int64 = 3
const Nz::Int64 = 2

input = zeros(Float32, Nx, Ny, Nz, 9) # T, P, Yi(NO AR)
input_cpu = zeros(Float32, Nx, Ny, Nz, 9) # T, P, Yi(NO AR)

# Lux.jl
mech = "./air.yaml"
ct = pyimport("cantera")
gas = ct.Solution(mech)
gas.TPY = 5500, 3679, "N2:77 O2:23"
input[:, :, :, 1] .= gas.T
input[:, :, :, 2] .= gas.P
for k ∈ 1:Nz, j ∈ 1:Ny, i ∈ 1:Nx
    input[i, j, k, 3:end] .= gas.Y[1:7]
end

input = CuArray(input)

j = JSON.parsefile("norm.json")
dt = 2e-8
lambda = j["lambda"]
inputs_mean = CuArray(convert(Vector{Float32}, j["inputs_mean"]))
inputs_std =  CuArray(convert(Vector{Float32}, j["inputs_std"]))
labels_mean = CuArray(convert(Vector{Float32}, j["labels_mean"]))
labels_std =  CuArray(convert(Vector{Float32}, j["labels_std"]))

inputs_norm = CUDA.zeros(Float32, 9, Nx*Ny*Nz)

function pre_input(input, inputs_norm, lambda, inputs_mean, inputs_std, Nx, Ny, Nz)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1)* blockDim().z + threadIdx().z

    if i > Nx || j > Ny || k > Nz
        return
    end

    for n = 3:9
        input[i, j, k, n] = (input[i, j, k, n]^lambda - 1) / lambda
    end

    for n = 1:9
        inputs_norm[n, i+Nx*((j-1) + (Ny * (k-1)))] = (input[i, j, k, n] - inputs_mean[n]) / inputs_std[n]
    end
    return
end

nthreads = (8, 8, 4)
nblock = (cld(Nx, 8), 
          cld(Ny, 8),
          cld(Nz, 4))

@cuda blocks=nblock threads=nthreads pre_input(input, inputs_norm, lambda, inputs_mean, inputs_std, Nx, Ny, Nz)

@time yt_pred = Lux.apply(model, dev_gpu(inputs_norm), ps, st)[1]
@. yt_pred = yt_pred * labels_std + labels_mean
yt_pred = yt_pred |> cpu_device()


y_pred = yt_pred[1:7, :]
t_pred = @. yt_pred[8, :] * dt * gas.T + gas.T
t_pred = reshape(t_pred, (Nx, Ny, Nz))
y_pred = reshape(y_pred, (7, Nx, Ny, Nz))

pred = zeros(Float64, Nx, Ny, Nz, 8)
copyto!(input_cpu, input)
for k ∈ 1:Nz, j ∈ 1:Ny, i ∈ 1:Nx
    pred[i, j, k, 1] = t_pred[i, j, k]
    pred[i, j, k, 2:end] = @. (lambda * (y_pred[:, i, j, k] * dt + input_cpu[i, j, k, 3:end]) + 1) ^ (1/lambda)
end