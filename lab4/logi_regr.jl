using Printf

abstract type OptMethod end

struct SteepestGD <: OptMethod end
struct Newton <: OptMethod end

Base.show(io::IO, optm::Newton) = print(io, "Newton's method")
Base.show(io::IO, optm::SteepestGD) = print(io, "Gradient descent")

σ(x) = one(x) / (one(x) + exp(-x))

mutable struct LogisticRegression
    Num::Int #number of feature
    θ::Array{Float64}
    optm::OptMethod
end

@inline bias(x) = vcat(ones(eltype(x), size(x, 2))', x)

LogisticRegression(n::Int, optm = SteepestGD()) = LogisticRegression(n, randn(n+1)', optm)

(lr::LogisticRegression)(x) = σ.(lr.θ * bias(x))

classify(lr::LogisticRegression, x) = 1 .- round.(lr(x))

function gradient(lr::LogisticRegression, x, y)
    p = lr(x)
    x = bias(x)
    - (1 .- y .- p) * x'
end

function hessian(lr::LogisticRegression, x, y)
    p = lr(x)
    x = bias(x)
    (x .* (p .* (1 .- p))) * x'
end

update_term(lr::LogisticRegression, x, y) = update_term(lr, x, y, lr.optm)

function update_term(lr::LogisticRegression, x, y, nt::Newton)
    H = hessian(lr, x, y)
    if iszero(det(H))
        update_term(lr, x, y, SteepestGD())
    else
        h = gradient(lr, x, y) * inv(H)'
        - h ./ norm(h)
    end
end

function update_term(lr::LogisticRegression, x, y, ::SteepestGD)
    g = gradient(lr, x, y)
    - g ./ norm(g)
end

function loglikelihood(θ, x, y)
    z = θ * bias(x)
    sum(log.(1 .+ exp.(-z)) .+ y .* z)
end

function fit!(lr::LogisticRegression, x, y; ϵ = 1e-6)
    i = 0
    θₙ = lr.θ
    while (θₙ₊₁ = θₙ + update_term(lr, x, y); abs2(loglikelihood(θₙ₊₁, x, y) - loglikelihood(θₙ, x, y)) > ϵ && i < 100000)
        θₙ .= θₙ₊₁
        i += 1
    end
    lr
end

function confusion(lr::LogisticRegression, x, y)
    c = classify(lr, x)
    #compute confusion matrix cfm
    TP = FP = TN = FN = 0
    for (pred, label) ∈ zip(c, y)
        if iszero(label)
            if iszero(pred)
                TP += 1
            else
                FN += 1
            end
        else
            if iszero(pred)
                FP += 1
            else
                TN += 1
            end
        end
    end
    cfm = [TP FN; FP TN]
    cfm
end

sensitivity(cfm) = cfm[1, 1] / (cfm[1, 1] + cfm[1, 2])
specificity(cfm) = cfm[2, 2] / (cfm[2, 2] + cfm[2, 1])

Base.copy(lr::LogisticRegression) = copy(lr, lr.optm)
Base.copy(lr::LogisticRegression, optm::OptMethod) = LogisticRegression(lr.Num, copy(lr.θ), optm)

function display_summary(lr::LogisticRegression, x, y)
    cfm = confusion(lr, x, y)

    print(lr.optm)
    println(":\n")
    println("w:")
    Base.print_array(stdout, lr.θ')
    println("\n")
    println("Confusion Matix:")
    println("             Predict cluster 1 Predict cluster 2")
    Printf.@printf "Is cluster 1 %9d %17d\n" cfm[1, 1] cfm[1, 2]
    Printf.@printf "Is cluster 2 %9d %17d\n" cfm[2, 1] cfm[2, 2]
    print("\n")
    println("Sensitivity (Successfully predict cluster 1): $(sensitivity(cfm))")
    println("Specificity (Successfully predict cluster 2): $(specificity(cfm))")
    print("\n")
end
