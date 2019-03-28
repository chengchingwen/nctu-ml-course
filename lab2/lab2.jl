using ArgParse

include("mnist.jl")
include("naive_bayes.jl")
include("online_learning.jl")

function parse_cmd()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--a"
            help = "beta param a for lab2-2"
            arg_type = Int
            default = 0
        "--b"
            help = "beta param b for lab2-2"
            arg_type = Int
            default = 0
        "--mode"
            help = "0-discrete/1-continuous mode for lab2-1"
            arg_type = Int
            default = 0
        "--datafile"
            help = "sample datas for lab2-2"
            arg_type = String
            default = joinpath(dirname(@__FILE__), "testfile.txt")
        "task"
            help = "1 for lab2-1 / 2 for lab2-2"
            arg_type = Int
            required = true
    end
    parse_args(s)
end

function display_bayes(p, label)
    println("Posterior (in log scale):")
    for i = 0:9
        println("$i: $(p[i])")
    end
    pred = predict(p)
    println("Prediction: $pred, Ans: $label")
    println()
    pred
end

function display_imagine(images::Matrix{Int})
    println("Imagination of numbers in Bayesian classifier:\n")
    for c = 0:9
        println("$c:")
        binary_plot(@view images[:, c+1])
        println()
    end

end

function lab21(mode::Int = 0)
    prepare_data()
    train_images, train_labels = process_mnist(mnist_trainset())
    test_images, test_labels = process_mnist(mnist_testset())

    prior = label_prior(train_labels)
    if mode == 0
        likelihood = discrete_likelihood(train_images, train_labels)
    else
        likelihood = continuous_likelihood(train_images, train_labels)
    end

    err = 0
    for (test_image, test_label) ∈ zip(test_images, test_labels)
        posterior = naive_bayes(likelihood, prior, test_image)
        pred = display_bayes(posterior, test_label)
        if pred != test_label
            err += 1
        end
    end

    imaginations = imagine(likelihood)
    display_imagine(imaginations)

    err_rate = err / length(test_labels)
    println("Error rate: $err_rate")

    err_rate
end

lab22(a, b,
      filename::AbstractString = joinpath(@__DIR__,
                                          "testfile.txt")) = open(filename) do fd
    samples = readlines(fd)
    lab22(a, b, samples)
end

function lab22(a, b, samples::Vector)
    bb = BetaBinomial(a, b)
    learn!(bb, samples)
end

function main()
    args = parse_cmd()
    task = args["task"]
    if task == 1
        mode = args["mode"]
        lab21(mode)
    elseif task == 2
        a = args["a"]
        b = args["b"]
        datafile = args["datafile"]
        lab22(a, b, datafile)
    else
        error("task not support")
    end
end

main()
