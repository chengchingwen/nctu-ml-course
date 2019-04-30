using Printf

abstract type OptMethod end

struct Newton <: OptMethod end
struct SteepestGD <: OptMethod end

Base.show(io::IO, optm::Newton) = print(io, "Newton's method")
Base.show(io::IO, optm::SteepestGD) = print(io, "Gradient descent")

σ(x) = one(x) / (one(x) + exp(-x))

struct LogisticRegression
    Num::Int #number of feature
    θ::Array{Float64}
    optm::OptMethod
end

LogisticRegression(n::Int, optm = SteepestGD()) = LogisticRegression(n, randn(n+1)', optm)

(lr::LogisticRegression)(x) = σ.(lr.θ * vcat(ones(eltype(x), size(x, 2))', x))

classify(lr::LogisticRegression, x) = round.(lr(x))

function fit!(lr::LogisticRegression, x, y)
    #TODO


end

function gradient(lr::LogisticRegression, x, y)
    z = lr(x)
    #TODO
end

function hessian(lr::LogisticRegression, x, y)
    z = lr(x)
    #TODO
end

function confusion(lr::LogisticRegression, x, y)
    c = classify(lr, x)
    #compute confusion matrix cfm
    #TODO
    #end
    cfm
end

Base.copy(lr::LogisticRegression) = LogisticRegression(lr.Num, copy(lr.θ), lr.optm)



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
    println("Sensitivity (Successfully predict cluster 1): $()") #TODO
    println("Specificity (Successfully predict cluster 2): $()") #TODO
    print("\n")
end

