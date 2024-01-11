using Plots
using JSON, PyCall
using Lux, JLD2

@load "luxmodel.jld2" model ps st

# Validation
# Call Cantera
mech = "./air.yaml"
TPX = 5320, 4000, "N2:77 O2:21 O:2 NO:1"
ct = pyimport("cantera")
ct_gas = ct.Solution(mech)
ct_gas.TPX = TPX
r = ct.IdealGasReactor(ct_gas, name="R1")
sim = ct.ReactorNet([r])
T_evo_ct = zeros(Float64, 10000)
Y_evo_ct = zeros(Float64, (5, 10000))
T_evo_ct[1] = ct_gas.T
Y_evo_ct[:, 1] = ct_gas.Y
dt = 5e-8

@time for i ∈ 1:9999
    sim.advance(i*dt)
    T_evo_ct[i+1] = ct_gas.T
    Y_evo_ct[:, i+1] = ct_gas.Y
end

# Lux.jl
gas = ct.Solution(mech)
gas.TPX = TPX
T_evo = zeros(Float64, 10000)
Y_evo = zeros(Float64, (5, 10000))
T_evo[1] = gas.T
Y_evo[:, 1] = gas.Y

input = zeros(Float64, 7)
j = JSON.parsefile("norm.json")
lambda = j["lambda"]
inputs_mean = convert(Vector{Float64}, j["inputs_mean"])
inputs_std = convert(Vector{Float64}, j["inputs_std"])
labels_mean = convert(Vector{Float64}, j["labels_mean"])
labels_std = convert(Vector{Float64}, j["labels_std"])

@time for i ∈ 1:9999
    input[1] = gas.T
    input[2] = gas.P
    input[3:end] = @. (gas.Y^lambda - 1) / lambda
    input_norm = @. (input - inputs_mean) / inputs_std
    yt_pred = Lux.apply(model, input_norm, ps, st)[1]
    @. yt_pred = yt_pred * labels_std + labels_mean
    y_pred = yt_pred[1:5]
    t_pred = yt_pred[6] * dt + gas.T
    @. y_pred = (lambda * (y_pred * dt + input[3:end]) + 1).^(1/lambda)
    gas.TDY = t_pred, gas.density, y_pred
    T_evo[i+1] = gas.T
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
p2 = plot(Y_evo', ls=:solid, lw = 2, lab=nothing)
p3 = plot(Y_evo_ct', ls=:solid, lw = 2, lab=nothing)

plot(p1, p2, p3, layout=@layout([a; b c]), size=(800,800))