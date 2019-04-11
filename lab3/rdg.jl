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

function Φ(x, n)
    ϕ = ones(n)
    @inbounds for j = 1:n-1
        ϕ[j+1] = x^j
    end
    ϕ
end

(pl::PolyLinear)(x) = pl.W' * Φ(x, pl.n)
generate(pl::PolyLinear) = pl(sample(Uniform(-10, 10))) + sample(pl.e)
