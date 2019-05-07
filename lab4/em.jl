struct BMM
    component_num::Int
    feature_num::Int
    μ::Matrix{Float64} # feature_num x component_num
    π::Vector{Float64} # component (x 1)
end

function BMM(k, d)
    mu = fill(0.5, d, k)
    BMM(k, d, mu)
end

function BMM(k, d, m)
    pi = rand(k)
    pi = pi ./ sum(pi)
    BMM(k, d, m, pi)
end

function likelihood(bmm::BMM, D) #D: f_num  x N
    N = length(D)
    logpi = log.(bmm.π)

    l = Vector{Vector{Float64}}(undef, N)
    # for (i, d) ∈ enumerate(D)
    Threads.@threads for i = 1:N
        d = @inbounds D[i]
        # d f x 1
        # mu f x k
        logp = d .* log.(bmm.μ) .+ (1 .- d) .* log.(1 .- bmm.μ) # f x k
        logp = reshape(sum(logp; dims=1), bmm.component_num) # k x 1
        logcp = logp .+ logpi
        cp = exp.(logcp)
        l[i] = cp ./ sum(cp) # k x 1
    end
    l
end

function ExpeMax!(bmm::BMM, D)
    z = expect(bmm, D)
    maximize!(bmm, z, D)
end

function expect(bmm::BMM, D)
    z = likelihood(bmm, D) # k x N
end

function maximize!(bmm::BMM, z, D)
    N = length(D)
    Nm = sum(z)

    μm = Vector{Array{Float64}}(undef, N)
    # for (i, (zi, d)) = enumerate(zip(z, D))
    Threads.@threads for i = 1:N
        zi = @inbounds z[i]
        d = @inbounds D[i]

        μ = d .* zi'
        μm[i] = μ
    end
    μm = sum(μm) ./ Nm'

    #clip
    μm = max.(μm, 1e-3)
    μm = min.(μm, 0.999)

    bmm.μ .= μm
    bmm.π .= Nm ./ N
    bmm
end

function predict(bmm::BMM, D)
    z = expect(bmm, D) #k x N
    l = map(argmax, z)
    l .- 1
end

function imagine(bmm::BMM)
    pix = map(x-> x > 0.5 ? 255 : 0, bmm.μ)
    pix'
end
