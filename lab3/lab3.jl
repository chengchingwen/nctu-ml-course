using ArgParse

include("./rdg.jl")
include("./seq_esti.jl")

function parse_cmd()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--m"
            help = "beta param a for lab3-1.a"
            arg_type = Float64
            default = 0.0
        "--s"
            help = "beta param b for lab3-1.a"
            arg_type = Float64
            default = 0.0
        "--n"
            help = "beta param a for lab3-1.b"
            arg_type = Float64
            default = 0.0
        "--a"
            help = "beta param a for lab3-1.b"
            arg_type = Float64
            default = 0.0
        "--w"
            help = "beta param a for lab3-1.b"
            arg_type = Float64
            action = :store_arg
            nargs = '*'
        "--datafile"
            help = "sample datas for lab2-2"
            arg_type = String
            default = joinpath(dirname(@__FILE__), "testfile.txt")
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
        length(w) != n && error("length(w) not equal to n")
        lab33(n, w, a)
    else
        error("task not support")
    end
end

#args = parse_cmd()
main()
