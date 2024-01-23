using Lux, LuxCUDA, JLD2, CUDA, LinearAlgebra

@load "luxmodel.jld2" model ps st
ps = ps |> gpu_device()
w1 = ps[1].weight
b1 = ps[1].bias
w2 = ps[2].weight
b2 = ps[2].bias
w3 = ps[3].weight
b3 = ps[3].bias

input = CUDA.ones(Float32, 7, 2048*1024)
Y1 = CUDA.ones(Float32, 64, 2048*1024)
Y2 = CUDA.ones(Float32, 256, 2048*1024)
Y3 = CUDA.ones(Float32, 6, 2048*1024)

# Zero GPU allocation
function evalModel(Y1, Y2, Y3, w1, w2, w3, b1, b2, b3, input)
    mul!(Y1, w1, input)
    Y1 .+= b1
    @. Y1 = gelu(Y1)

    mul!(Y2, w2, Y1)
    Y2 .+= b2
    @. Y2 = gelu(Y2)

    mul!(Y3, w3, Y2)
    Y3 .+= b3

    return Y3
end

# CUDA.@time for n=1:100
#     model(input, ps, st)[1]
# end

CUDA.@time for n=1:100
    evalModel(Y1, Y2, Y3, w1, w2, w3, b1, b2, b3, input)
end