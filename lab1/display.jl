using Plots

#xs, ys = read_sample("testfile.txt")
#scatterplot(xs, ys, title = "My Scatterplot")

function stringXn(n)
    n0 = n
    sd = Dict(0=>"⁰", 1=>"¹", 2=>"²", 3=>"³", 4=>"⁴",  5=>"⁵", 6=>"⁶", 7=>"⁷", 8=>"⁸", 9=>"⁹")
    sx = String[]
    while n != 0
        n, i = divrem(n, 10)
        push!(sx, sd[abs(i)])
    end

    if n0 == 0
        return ""
    elseif n0 < 0
        push!(sx, "⁻")
        push!(sx, "X")
        return join(Base.Iterators.reverse(sx))
    else
        push!(sx, "X")
        return join(Base.Iterators.reverse(sx))
    end
end

function display_ploy(x)
    for (i, p) ∈ enumerate(Base.Iterators.reverse(x))
        if p >= 0.0
            if i > 1
                print(" + ")
            end
        else
            if i > 1
                print(" - ")
            else
                print("-")
            end
        end
        print("$(abs(p))$(stringXn(length(x) - i))")
    end
    println()
end

function display_result(name, x, xs, ys)
    println("$name:")
    print("Fitting line: ")
    display_ploy(x)
    error = Terror(xs, ys, x)
    println("Total error: $error")
end

function plot_result(name, x, xs, ys)
    lb, ub = minimum(xs)-1, maximum(xs)+1
    plt = plot(ploy(x), lb, ub)
    scatter!(plt, xs, ys, title = name)
end

