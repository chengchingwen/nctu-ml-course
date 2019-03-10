function Φ(xs, n)
    ϕ = ones(length(xs), n)
    for (i, x) ∈ enumerate(xs)
        @inbounds for j = 1:n-1
            ϕ[i, j+1] = x^j
        end
    end
    ϕ
end

function LSE(xs, ys, n; λ::Float64 = 0.0)
    ϕ = Φ(xs, n)
    A = ϕ' * ϕ
    inverse(A + λ * one(A)) * ϕ' * ys
end

function NewtonMethod(xs, ys, n; ϵ = 1e-6)
    ϕ = Φ(xs, n)
    A = ϕ' * ϕ
    Aᵗb = ϕ' * ys

    xₙ = randn(n)
    while sum(x->x^2, (xₙ₊₁ = xₙ - inverse(2A) * 2(A * xₙ - Aᵗb)) - xₙ) > ϵ
        xₙ = xₙ₊₁
    end
    xₙ₊₁
end

Terror(xs, ys, x) = sum(x->x^2, Φ(xs, length(x)) * x - ys)

function ploy(ps)
    function (x)
        s = 0.0
        for (i, p) ∈ enumerate(ps)
            s += p * x^(i-1)
        end
        s
    end
end
