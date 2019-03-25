import Base: +
using Statistics: std, mean

abstract type Distributation end

struct BinDistributation <: Distributation
    label_num::Int
    label_prob::Vector{Float64}
end

struct GaussianDistributation <: Distributation
    μ::Float64
    σ::Float64
end

prob(gd::GaussianDistributation, x) = inv(√(2π) * gd.σ) * ℯ ^ -((x - gd.μ)^2 / 2gd.σ^2)
logprob(gd::GaussianDistributation, x) = -log(√(2π) * gd.σ) - ((x - gd.μ)^2 / 2gd.σ^2)


BinDistributation(l::Int) = BinDistributation(l, zeros(l))
BinDistributation(l::Int, v::Float64) = BinDistributation(l, fill(v, l))

(bd::BinDistributation)(x::Int) = bd.label_prob[x+1]

update!(bd::BinDistributation, l; inc=1) = (bd.label_prob[l+1] += inc; bd)
scale!(bd::BinDistributation, s) = (bd.label_prob .*= s; bd)

prob(bd::BinDistributation, x) = bd.label_prob[div(x, 8)+1]
logprob(bd::BinDistributation, x) = log(prob(bd, x))

Base.sum(bd::BinDistributation) = sum(bd.label_prob)
Base.getindex(bd::BinDistributation, i) = bd.label_prob[i+1]
Base.argmax(bd::BinDistributation) = argmax(bd.label_prob)
Base.argmin(bd::BinDistributation) = argmin(bd.label_prob)

function +(bd1::BinDistributation, bd2::BinDistributation)
    @assert bd1.label_num == bd2.label_num
    BinDistributation(bd1.label_num, bd1.label_prob .+ bd2.label_prob)
end

function label_prior(labels::Vector{Int})
    p = BinDistributation(10)
    total = Float64(length(labels))
    for l ∈ labels
        update!(p, l)
    end

    scale!(p, inv(total))
end

function discrete_likelihood(images::Vector{Vector{Int}}, labels::Vector{Int}, image_size=(28,28))
    p = [[BinDistributation(32, 0.1) for i = 1:prod(image_size)] for j = 1:10]

    for (image, label) ∈ zip(images, labels)
        for (i, x) ∈ enumerate(image)
            update!(p[label+1][i], div(x, 8))
        end
    end

    for pi ∈ p
        for d ∈ pi
            scale!(d, inv(sum(d)))
        end
    end

    p
end

function continuous_likelihood(images::Vector{Vector{Int}}, labels::Vector{Int}, image_size=(28,28))
    p = Vector{Vector{GaussianDistributation}}(undef, 10)
    for c = 1:10
        mat = hcat(images[labels .== c-1]...)
        μs = mean(mat; dims=2)
        σs = std(mat; dims=2) .+ 0.01
        gs = map((μ, σ)->GaussianDistributation(μ, σ), μs, σs)
        p[c] = reshape(gs, :)
    end

    p
end

function naive_bayes(likelihood::Vector{Vector{D}}, prior::BinDistributation, x) where D <: Distributation
    ap = log.(prior.label_prob)
    for c = 1:10
        for (pi, xi) ∈ zip(likelihood[c], x)
            lnpi_xgc = logprob(pi, xi)
            ap[c] += lnpi_xgc
        end
    end

    BinDistributation(10, ap ./ sum(ap))
end

predict(bd::BinDistributation) = argmin(bd) - 1

function imagine(likelihood::Vector{Vector{BinDistributation}}, c)
    image = zeros(Int, 784)
    for (i, pi) ∈ enumerate(likelihood[c+1])
        darkp = sum(pi.label_prob[1:15])
        lightp = sum(pi.label_prob[16:end])
        if darkp > lightp
            image[i] = 0
        else
            image[i] = 255
        end
    end
    image
end

function imagine(likelihood::Vector{Vector{GaussianDistributation}}, c)
    image = zeros(Int, 784)
    for (i, pi) ∈ enumerate(likelihood[c+1])
        darkp = sum(x->prob(pi, x), 0:127)
        lightp = sum(x->prob(pi, x), 128:255)
        if darkp > lightp
            image[i] = 0
        else
            image[i] = 255
        end
    end
    image
end

function imagine(likelihood::Vector{Vector{D}}) where D <: Distributation
    images = zeros(Int, (784, 10))
    for i = 0:9
        (@view images[:, i+1]) .= imagine(likelihood, i)
    end
    images
end
