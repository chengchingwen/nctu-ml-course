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

