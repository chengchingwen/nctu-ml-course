using ArgParse
using Plots

include("../lab3/rdg.jl")
includet("logi_regr.jl")
# include("../lab2/mnist.jl")
# include("em.jl")

function parse_cmd()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--N"
            help = "number of data points"
            arg_type = Int
            default = 0
        "--mx1"
            help = "x1 mean"
            arg_type = Float64
            default = 0.0
        "--vx1"
            help = "x1 variance"
            arg_type = Float64
            default = 1.0
        "--my1"
            help = "y1 mean"
            arg_type = Float64
            default = 0.0
        "--vy1"
            help = "y1 variance"
            arg_type = Float64
            default = 1.0
        "--mx2"
            help = "x2 mean"
            arg_type = Float64
            default = 0.0
        "--vx2"
            help = "x2 variance"
            arg_type = Float64
            default = 1.0
        "--my2"
            help = "y2 mean"
            arg_type = Float64
            default = 0.0
        "--vy2"
            help = "y2 variance"
            arg_type = Float64
            default = 1.0
        "task"
            help = "1 for lab4-1 / 2 for lab4-2"
            arg_type = Int
            required = true
    end
    parse_args(s)
end

sample(dist::D, N) where D = hcat([sample(dist) for i = 1:N]...)

function plot_result(m1, m2, x, y)
    N = Int(length(y) / 2)
    g = scatter(x[1,1:N], x[2,1:N], color=:blue, title="Ground truth")
    scatter!(g, x[1,N+1:end], x[2,N+1:end], color=:red)

    p1 = plot_result(string(m1.optm), m1, x)
    p2 = plot_result(string(m2.optm), m2, x)

    plot(g, p1, p2, layout=(1, 3), legend=false)
end

function plot_result(title, model, x)
    p = classify(model, x)
    b = map(first, filter(x->iszero(x[2]), collect(enumerate(p))))
    r = map(first, filter(x->isone(x[2]), collect(enumerate(p))))
    plt = scatter(x[1,b], x[2,b], color=:blue, title=title, legend=false)
    scatter!(plt, x[1,r], x[2,r], color=:red, legend=false)

    plt
end

function lab41(N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2)
    D1dist = MvNormal([mx1, my1], [vx1 zero(vx1); zero(vy1) vy1])
    D2dist = MvNormal([mx2, my2], [vx2 zero(vx2); zero(vy2) vy2])
    D1 = sample(D1dist, N) # label 0
    L1 = zeros(Float64, N)'
    D2 = sample(D2dist, N) # label 1
    L2 = ones(Float64, N)'
    D = hcat(D1, D2)
    L = hcat(L1, L2)

    model_sgd = LogisticRegression(2, SteepestGD())
    model_nt = copy(model_sgd, Newton())

    fit!(model_sgd, D, L)
    fit!(model_nt,  D, L)

    display_summary(model_sgd, D, L)
    println("----------------------------------------")
    display_summary(model_nt , D, L)
    plot_result(model_sgd, model_nt, D, L)
end

function lab42()

end


function main()
    args = parse_cmd()
    task = args["task"]
    if task == 1
        N = args["N"]
        mx1 = args["mx1"]
        vx1 = args["vx1"]
        my1 = args["my1"]
        vy1 = args["vy1"]
        mx2 = args["mx2"]
        vx2 = args["vx2"]
        my2 = args["my2"]
        vy2 = args["vy2"]
        lab41(N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2)
    elseif task == 2
        lab42()
    else
        error("task not support")
    end
end
