struct BayesLinear
    n::Int
    b::Float64
    prior::MvNormal
    _seenx::Vector{Float64}
    _seeny::Vector{Float64}
    a::Float64
    function BayesLinear(n::Int, b::Float64, a::Float64)
        prior = MvNormal(zeros(n), inv(b) * one(randn(n, n)))

        new(n, b, prior, Float64[], Float64[], a)
    end
end

function update!(bl::BayesLinear, x, y)
    push!(bl._seenx, x)
    push!(bl._seeny, y)
    bl
end

function posterior(bl::BayesLinear)
    ϕ = Φ(bl._seenx, bl.n)

    Sn = inv(bl.b*I + bl.a * ϕ' * ϕ)
    m = bl.a * Sn * ϕ' * bl._seeny

    MvNormal(m, Sn)
end

function predictive(bl::BayesLinear, x)
    pd = posterior(bl)
    ϕ = Φ(x, bl.n)

    m = pd.μ' * ϕ
    s = inv(bl.a) + ϕ' * pd.Λ * ϕ
    Normal(m, s)
end

function pred_func(bl::BayesLinear)
    pd = posterior(bl)
    n = bl.n
    a = bl.a

    function d(x)
        ϕ = Φ(x, n)

        m = pd.μ' * ϕ
        s = inv(a) + ϕ' * pd.Λ * ϕ
        m, √s
    end

    x->first(d(x)), x->+(d(x)...), x->-(d(x)...)
end

function pred_func(pl::PolyLinear)
    σ = √pl.e.σ²
    x->pl(x), x->pl(x)+σ, x->pl(x)-σ
end
