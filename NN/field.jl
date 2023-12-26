using Plots, CUDA
using JSON, PyCall
using Lux, LuxCUDA, JLD2

dev_cpu = cpu_device()
dev_gpu = gpu_device()


@load "luxmodel.jld2" model ps st

ps = ps |> dev_gpu
st = st |> dev_gpu

N::Int64 = 512

input = zeros(Float64, N, N, 9) # T, P, Yi(NO AR)
input_cpu = zeros(Float64, N, N, 9) # T, P, Yi(NO AR)

# Lux.jl
mech = "./air.yaml"
ct = pyimport("cantera")
gas = ct.Solution(mech)
gas.TPY = 350, 3596, "N2:77 O2:23"
input[:, :, 1] .= gas.T
input[:, :, 2] .= gas.P
for j ∈ 1:N, i ∈ 1:N
    input[i, j, 3:end] .= gas.Y[1:7]
end

input = CuArray(input)

j = JSON.parsefile("norm.json")
dt = 1e-6
lambda = j["lambda"]
inputs_mean = CuArray(convert(Vector{Float64}, j["inputs_mean"]))
inputs_std =  CuArray(convert(Vector{Float64}, j["inputs_std"]))
labels_mean = CuArray(convert(Vector{Float64}, j["labels_mean"]))
labels_std =  CuArray(convert(Vector{Float64}, j["labels_std"]))

inputs_norm = CUDA.zeros(Float64, 9, N*N)

function pre_input(input, inputs_norm, lambda, inputs_mean, inputs_std, N)
    i = (blockIdx().x-1)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1)* blockDim().y + threadIdx().y

    if i > N || j > N
        return
    end

    for n = 3:9
        input[i, j, n] = (input[i, j, n]^lambda - 1) / lambda
    end

    for n = 1:9
        inputs_norm[n, (i-1)*N+j] = (input[i, j, n] - inputs_mean[n]) / inputs_std[n]
    end
    return
end

nthreads = (16, 16, 1)
nblock = (cld(N, 16), 
          cld(N, 16),
          1)

@cuda blocks=nblock threads=nthreads pre_input(input, inputs_norm, lambda, inputs_mean, inputs_std, N)

@time yt_pred = Lux.apply(model, dev_gpu(inputs_norm), ps, st)[1]
@. yt_pred = yt_pred * labels_std + labels_mean
yt_pred = yt_pred |> cpu_device()


y_pred = yt_pred[1:7, :]
@. t_pred = yt_pred[8, :] * dt * gas.T + gas.T
t_pred = reshape(t_pred, (N, N))
y_pred = reshape(y_pred, (7, N, N))

pred = zeros(Float64, N, N, 8)
copyto!(input_cpu, input)
for j ∈ 1:N, i ∈ 1:N
    pred[i, j, 1] = t_pred[i, j]
    pred[i, j, 2:end] = @. (lambda * (y_pred[:, i, j] * dt + input_cpu[i, j, 3:end]) + 1) ^ (1/lambda)
end