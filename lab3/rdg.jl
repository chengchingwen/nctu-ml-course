using LinearAlgebra

struct Uniform
    a::Float64
    b::Float64
    Uniform(a, b) = a > b ? error("b should >= a") : new(a, b)
end

sample(U::Uniform) = rand() * (U.b - U.a) + U.a

struct Normal
    μ::Float64
    σ²::Float64
end

sample(N::Normal) = (√(-2log(rand())) * cos(2π*rand()) * √(N.σ²)) + N.μ

struct MvNormal
    μ::Vector{Float64}
    Λ::Matrix{Float64}
end

sample(MN::MvNormal) = cholesky(MN.Λ).L * [sample(Normal(0.0, 1.0)) for i = 1:length(MN.μ)] + MN.μ

struct PolyLinear
    n::Int
    W::Vector{Float64}
    e::Normal
    function PolyLinear(n::Int, w::Vector{<:Real}, a::N) where N <: Real
        length(w) != n && error("length(w) not equal to n")

        e = Normal(0, a)
        new(n, w, e)
    end
end

function Φ(x::Float64, n)
    ϕ = ones(n)
    @inbounds for j = 1:n-1
        ϕ[j+1] = x^j
    end
    ϕ
end

function Φ(xs::Array, n)
    ϕ = ones(length(xs), n)
    for (i, x) ∈ enumerate(xs)
        @inbounds for j = 1:n-1
            ϕ[i, j+1] = x^j
        end
    end
    ϕ
end

(pl::PolyLinear)(x) = pl.W' * Φ(x, pl.n)
function generate(pl::PolyLinear, bound::Float64=10.0)
    x = sample(Uniform(-bound, bound))
    y = pl(x) + sample(pl.e)
    x, y
end
