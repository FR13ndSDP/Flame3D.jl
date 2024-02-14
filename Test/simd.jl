using BenchmarkTools

a = rand(Float64, 2048, 2048)

function test(a)
    sum = zero(eltype(a))
    for i in eachindex(a)
        sum += a[i]
    end
    return sum
end

function test0(a)
    sum = zero(eltype(a))
    for j in 1:2048
        for i in 1:2048
            @inbounds sum += a[i, j]
        end
    end
    return sum
end


function test1(a)
    sum = zero(eltype(a))
    @inbounds for i in eachindex(a)
        sum += a[i]
    end
    return sum
end

function test2(a)
    sum = zero(eltype(a))
    @inbounds @simd for i in eachindex(a)
        sum += a[i]
    end
    return sum
end
