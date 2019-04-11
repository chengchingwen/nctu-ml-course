mutable struct BayesLinear
    n::Int
    b::Float64
    prior::MvNormal
    a::Float64
    function BayesLinear(n::Int, b::Float64, a::Float64)
        prior = MvNormal(zeros(n), inv(b) * one(randn(n, n)))

        new(n, b, prior, a)
    end
end

function update!(bl::BayesLinear, x, y)
    ϕ = Φ(x, bl.n)
    sp = inv(bl.prior.Λ)
    s = inv(bl.a * ϕ * ϕ' + sp)
    mu = bl.a * s * (ϕ * y + inv(bl.a) * sp * bl.prior.μ)

    bl.prior = MvNormal(mu, s)
    bl
end

function posterior(bl::BayesLinear)
    bl.prior
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
