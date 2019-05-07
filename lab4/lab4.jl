using Printf
using Random

using ArgParse
using Plots

include("../lab3/rdg.jl")
include("logi_regr.jl")
include("../lab2/mnist.jl")
include("em.jl")

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

function image_preprocess(images)
    images = map(images) do image
        for i = 1:length(image)
            v = image[i]::Int
            if 0 <= v <= 127
                image[i] = 0
            else
                image[i] = 1
            end
        end
        image
    end
    images
end

function label_dist(images, labels)
    c = zeros(Int, 10)
    d = zeros(Int, 784, 10)

    for (i, l) ∈ zip(images, labels)
        c[l+1] +=1
        d[:, l+1] .+= i
    end

    d ./ c'
end

normd1(x) =  x ./ (sum(x .^ 2; dims=1) .^ 0.5)

# function relabel(bmm::BMM, label_dists)
#     reshape(map(x->x[1]-1, argmax(normd1(bmm.μ)' * normd1(label_dists);dims=1)), :)
# end

function tonewlabel(bmm::BMM, label_dists)
    reshape(map(x->x[1]-1, argmax((normd1(bmm.μ)' * normd1(label_dists))';dims=1)), :)
end


function plot_imagine(imagines, newlabels; final=false)
    for i = 1:10
        if final
            println("labeled class $(newlabels[i])")
            binary_plot(imagines[i, :])
        else
            println("class $(i-1)")
            binary_plot(imagines[i, :])
        end
    end
end

function confusions(preds, labels)
    cfm = zeros(Int, 2, 2, 10)
    for (pred, label) ∈ zip(preds, labels)
        for i = 1:10
            if label == i-1
                if pred == label
                    cfm[1,1,i] += 1
                else
                    cfm[1,2,i] += 1
                end
            else
                if pred == label
                    cfm[2,1,i] += 1
                else
                    cfm[2,2,i] += 1
                end
            end
        end
    end
    cfm
end

function plot_confusion(cfms)
    for i = 0:9
        cfm = cfms[:,:,i+1]
        println("Confusion Maxtix $(i):")
        println("             Predict number $i Predict not number $i")
        Printf.@printf "Is number %d %9d %17d\n" i cfm[1, 1] cfm[1, 2]
        Printf.@printf "Is not number %d %9d %17d\n" i cfm[2, 1] cfm[2, 2]
        print("\n")
        println("Sensitivity (Successfully predict number $i): $(sensitivity(cfm))")
        println("Specificity (Successfully predict not $i): $(specificity(cfm))")
        print("\n")
        println("------------------------------------------------------------")
        print("\n")
    end
end

function stat_init(D)
    N = length(D)
    perN = N ÷ 10
    mu0 = Array{Float64}(undef, 784, 10)
    for i = 1:10
        splt = D[(1 + (i-1)*perN):(i*perN)]
        mu0[:, i] .= sum(splt) ./ perN
    end
    mu0
end

function lab42(;ϵ = 1e-6)
    i = 0
    timages, tlabels = process_mnist(mnist_trainset())
    timages = image_preprocess(timages)

    μ0 = stat_init(sort(timages, by=x->count(isequal(1), x)+rand(1:50)))
    bmm = BMM(10, 784, μ0 .+ 1e-6)
    zₙ = expect(bmm, timages)

    while (obj = sum(x->sum(abs2, x), (zₙ₊₁ = expect(ExpeMax!(bmm, shuffle(timages)), timages)) - zₙ)) > ϵ
        i+=1
        zₙ = zₙ₊₁
        plot_imagine(imagine(bmm), 1:10)
        @info "Iteration: $i, Difference: $obj"
        println("------------------------------------------------------------")
    end

    ld = label_dist(timages, tlabels)
    newlabels = tonewlabel(bmm, ld)
    pred = predict(bmm, timages)
    pred = map(x->newlabels[x+1], pred)

    plot_imagine(imagine(bmm), newlabels; final=true)
    cfms = confusions(pred, tlabels)
    plot_confusion(cfms)

    err = sum(tlabels .!= pred) / length(pred)

    println("Total iteration to converge: $i")
    println("Total error rate: $err")
    bmm
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
