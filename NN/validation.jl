using Plots
using JSON, PyCall
using Lux, JLD2

@load "luxmodel.jld2" model ps st

# Validation
# Call Cantera
mech = "./air.yaml"
TPY = 3000, 0.04*101325, "N2:77 O2:23"
ct = pyimport("cantera")
ct_gas = ct.Solution(mech)
ct_gas.TPY = TPY
r = ct.IdealGasReactor(ct_gas, name="R1")
sim = ct.ReactorNet([r])
T_evo_ct = zeros(Float64, 10000)
P_evo_ct = zeros(Float64, 10000)
Y_evo_ct = zeros(Float64, (8, 10000))
T_evo_ct[1] = ct_gas.T
P_evo_ct[1] = ct_gas.P
Y_evo_ct[:, 1] = ct_gas.Y

@time for i ∈ 1:9999
    sim.advance(i*1e-8)
    T_evo_ct[i+1] = ct_gas.T
    P_evo_ct[i+1] = ct_gas.P
    Y_evo_ct[:, i+1] = ct_gas.Y
end

# Lux.jl
gas = ct.Solution(mech)
gas.TPY = TPY
T_evo = zeros(Float64, 10000)
P_evo = zeros(Float64, 10000)
Y_evo = zeros(Float64, (8, 10000))
T_evo[1] = gas.T
P_evo[1] = gas.P
Y_evo[:, 1] = gas.Y

input = zeros(Float64, 9)
j = JSON.parsefile("norm.json")
dt = 1e-8
lambda = j["lambda"]
inputs_mean = convert(Vector{Float64}, j["inputs_mean"])
inputs_std = convert(Vector{Float64}, j["inputs_std"])
labels_mean = convert(Vector{Float64}, j["labels_mean"])
labels_std = convert(Vector{Float64}, j["labels_std"])

@time for i ∈ 1:9999
    input[1] = gas.T
    input[2] = gas.P
    input[3:end] = @. (gas.Y[1:7]^lambda - 1) / lambda
    input_norm = @. (input - inputs_mean) / inputs_std
    yt_pred = Lux.apply(model, input_norm, ps, st)[1]
    @. yt_pred = yt_pred * labels_std + labels_mean
    y_pred = yt_pred[1:7]
    t_pred = yt_pred[8] * dt * gas.T + gas.T
    @. y_pred = (lambda * (y_pred * dt + input[3:end]) + 1).^(1/lambda)
    append!(y_pred, gas.Y[end])
    gas.TDY = t_pred, gas.density, y_pred
    T_evo[i+1] = gas.T
    P_evo[i+1] = gas.P
    Y_evo[:, i+1] = gas.Y
end

# compute relative error
Err = @. abs(Y_evo - Y_evo_ct)/(Y_evo_ct + 1f-20)
max_err = [maximum(c) for c in eachslice(Err, dims=1)]
println("Max relative error for Y: $max_err")

Err = @. abs(T_evo - T_evo_ct)/(T_evo_ct + 1f-20)
max_err = maximum(Err)
println("Max relative error for T: $max_err")

# fig
gr()
p1 = plot([T_evo T_evo_ct], w = 1, lab = ["predict-T" "cantera-T"], ls=[:dot :solid], lw = 2)
p2 = plot([P_evo P_evo_ct], w = 1, lab = ["predict-P" "cantera-P"], ls=[:dot :solid], lw = 2)
p3 = plot(Y_evo[:, :]', ls=:solid, lw = 2, lab=nothing)
p3 = plot!(Y_evo_ct[:, :]', ls=:dot, lw = 2, lab=nothing)

plot(p1, p2, p3, layout=@layout([a; b; c]), size=(800,800))