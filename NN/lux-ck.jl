using Lux, LuxCUDA, Optimisers, Random, Statistics, Zygote, DelimitedFiles, MLUtils, JLD2
using Plots

# Backend of Plots
unicodeplots()

dev_cpu = cpu_device()
dev_gpu = gpu_device()

# Configs
train_epoch = 2000
batch_size = 1024
lr = 1f-3
decay_rate = 0.3
decay_step = 500
epoch_out = 201

inputs = readdlm("input.txt", Float32)
labels = readdlm("label.txt", Float32)
inputs = inputs'
labels = labels'
data_size = size(inputs)[2]

function get_dataloaders(; dataset_size=data_size, sequence_length=50)
    # Split the dataset
    # (x_train, y_train), (x_val, y_val) = splitobs((inputs, labels); at=0.8, shuffle=true)
    # Create DataLoaders
    return (
        # Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((inputs, labels)); batchsize=batch_size, shuffle=true))
        # Don't shuffle the validation data
        # DataLoader(collect.((x_val, y_val)); batchsize=8192, shuffle=false))
end

rng = MersenneTwister()
Random.seed!(rng, 12345)

model = Lux.Chain(Lux.Dense(7 => 64, gelu), 
                  Lux.Dense(64 => 256, gelu), 
                  Lux.Dense(256 => 6))

opt = Optimisers.Adam(lr)

function loss_function(model, ps, st, data)
    y_pred, st = Lux.apply(model, data[1], ps, st)
    mse_loss = mean(abs2, y_pred .- data[2])
    return mse_loss, st, ()
end

tstate = Lux.Training.TrainState(rng, model, opt)

vjp_rule = Lux.Training.AutoZygote()

# define a scheduler
function exp_scheduler(epoch, decay_step, lr, decay_rate)
	lr *= decay_rate^(cld(epoch, decay_step)-1)
	return lr
end

loss_all = Float64[]
function main(tstate::Lux.Experimental.TrainState, vjp, epochs)
    train_loader = get_dataloaders()
    for epoch in 1:epochs
        # get batch data
        loss_sum = 0
        @time for data_batch in train_loader
            data_batch = data_batch |> gpu_device()
            grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp,
                loss_function, data_batch, tstate)

            tstate = Lux.Training.apply_gradients(tstate, grads)
            loss_sum += loss
        end
        push!(loss_all, loss_sum/floor(Int, data_size/batch_size))
        print("Epoch: ")
        printstyled("$(epoch)"; color=:blue, bold=true)
        print(" || Loss: ")
        printstyled("$(loss_all[epoch])\n"; color=:red)
        Optimisers.adjust!(tstate.optimizer_state, exp_scheduler(epoch, decay_step, lr, decay_rate))

        if (epoch % 10 == 0)
            plot(loss_all, yscale=:log10, lw = 2, show=true, lab="loss")
        end

        if epoch % epoch_out == 0 || epoch == epochs
            ps = tstate.parameters |> dev_cpu
            st = tstate.states |> dev_cpu
            @save "luxmodel.jld2" model ps st
        end
    end
    return
end

main(tstate, vjp_rule, train_epoch)
