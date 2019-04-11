import Statistics: mean, var

mutable struct Welford
    M₂ₙ::Float64
    Mₙ::Float64
    n::Int
end

Welford() = Welford(0.0, 0.0, 0)

function update!(w::Welford, xₙ)
    w.n += 1
    x̄ₙ₋₁ = w.Mₙ
    x̄ₙ = x̄ₙ₋₁ + (xₙ - x̄ₙ₋₁) / w.n

    w.Mₙ = x̄ₙ
    w.M₂ₙ += (xₙ - x̄ₙ₋₁) * (xₙ - x̄ₙ)
    w
end

mean(w::Welford) = w.Mₙ
var(w::Welford) = w.M₂ₙ / w.n
