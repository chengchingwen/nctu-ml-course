function read_sample(filename::AbstractString)
    open(filename) do f
        xs = Float64[]
        ys = Float64[]
        for l ∈ eachline(f)
            x, y = split(l, ",")
            push!(xs, parse(Float64, x))
            push!(ys, parse(Float64, y))
        end
        xs, ys
    end
end


function gen_sample(f, n)
    xs = 6 * randn(n)
    ys = f.(xs) .+ randn()
    xs, ys
end
