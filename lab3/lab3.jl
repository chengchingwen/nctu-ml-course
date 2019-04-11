using ArgParse
using Plots

include("./rdg.jl")
include("./seq_esti.jl")
#include("./bayes_regr.jl")
include("./bayes_online.jl")
include("./plot.jl")

function parse_cmd()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--m"
            help = "mean for lab3-1.a"
            arg_type = Float64
            default = 0.0
        "--s"
            help = "variance for lab3-1.a"
            arg_type = Float64
            default = 1.0
        "--n"
            help = "n basis for lab3-1.b"
            arg_type = Int
            default = 0
        "--a"
            help = "noies variance for lab3-1.b"
            arg_type = Float64
            default = 0.0
        "--w"
            help = "weight w for lab3-1.b"
            arg_type = Float64
            action = :store_arg
            nargs = '*'
        "--b"
            help = "precision b for lab3-3"
            arg_type = Float64
            default = 1.0
        "task"
            help = "1 for lab3-1 / 2 for lab3-2 / 3 for lab3-3"
            arg_type = Int
            required = true
    end
    parse_args(s)
end


function isconverge(w::Welford, μₙ₋₁, σ²ₙ₋₁, ϵ = 1e-6)
    w.n == 0 && return false

    μₙ = mean(w)
    σ²ₙ = var(w)

    return (μₙ - μₙ₋₁)^2 <= ϵ && (σ²ₙ - σ²ₙ₋₁)^2 <= ϵ
end


function lab32(m, s)
    println("Data point source function: N($m, $s)\n")

    N = Normal(m, s)
    w = Welford()

    μₙ₋₁ = mean(w)
    σ²ₙ₋₁ = var(w)

    while !isconverge(w, μₙ₋₁, σ²ₙ₋₁)
        μₙ₋₁ = mean(w)
        σ²ₙ₋₁ = var(w)

        xₙ = sample(N)
        update!(w, xₙ)


        μ = mean(w)
        σ² = var(w)

        println("Add data point: $xₙ")
        println("Mean = $μ   Variance = $σ²")
    end

    w
end

function isconverge(MN::MvNormal, m, s, ϵ = 1e-9)
    sum(abs2, m) == 0 && return false

    mn = MN.μ
    sn = MN.Λ

    return sum(abs2, mn - m) <= ϵ && sum(abs2, sn - s) <= ϵ
end

function lab33(n, w, a, b)
    PL = PolyLinear(n, w, a)

    bl = BayesLinear(n, b, inv(a))

    m = bl.prior.μ
    s = bl.prior.Λ
    post = bl.prior

    fs = Any[]
    c = 0
    seenx = Float64[]
    seeny = Float64[]
    xs = Any[]
    ys = Any[]

    while !isconverge(post, m, s)
        m = post.μ
        s = post.Λ
        c+=1

        x, y = generate(PL, 1.0)

        push!(seenx, x)
        push!(seeny, y)

        update!(bl, x, y)
        post = posterior(bl)
        pred = predictive(bl, x)

        if c == 10
            push!(fs, pred_func(bl))
            push!(xs, copy(seenx))
            push!(ys, copy(seeny))
        elseif c == 50
            push!(fs, pred_func(bl))
            push!(xs, copy(seenx))
            push!(ys, copy(seeny))
        end

        println("Add data point ($x, $y):\n")
        println("\n")
        println("Posterior mean:")
        Base.print_array(stdout, post.μ)
        println("\n")
        println("Posterior variance:")
        Base.print_array(stdout, post.Λ)
        println("\n")
        println("Predictive distribution ~ N($(pred.μ), $(pred.σ²))")
        println("------------------------------------------------------")
    end

    pushfirst!(xs, copy(seenx))
    pushfirst!(ys, copy(seeny))
    pushfirst!(fs, pred_func(bl))
    pushfirst!(fs, pred_func(PL))

    display(plot_result(fs, xs, ys))

    bl, fs, xs, ys
end


function main()
    args = parse_cmd()
    task = args["task"]
    if task == 2
        m = args["m"]
        s = args["s"]
        lab32(m, s)
    elseif task == 3
        w = args["w"]
        n = args["n"]
        a = args["a"]
        b = args["b"]
        lab33(n, w, a, b)
    else
        error("task not support")
    end
end

#args = parse_cmd()
main()
