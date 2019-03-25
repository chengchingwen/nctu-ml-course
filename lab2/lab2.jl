include("mnist.jl")
include("naive_bayes.jl")


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
