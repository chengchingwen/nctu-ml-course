using ArgParse

include("inverse.jl")
include("fit.jl")
include("sample.jl")
include("display.jl")


function parse_cmd()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--lambda"
            help = "lambda for LSE regularization"
            arg_type = Float64
            default = 0.0
        "--datafile"
            help = "data as csv-format file"
            arg_type = String
            default = "testfile.txt"
        "n"
            help = "number of ploynomial bases"
            arg_type = Int
            required = true
    end
    parse_args(s)
end

function fitboth(xs, ys, n; λ::Float64 = 0.0)
    lsex = LSE(xs, ys, n; λ = λ)
    newtonx = NewtonMethod(xs, ys, n)

    display_result("LSE", lsex, xs, ys)
    println()
    display_result("Newton's Method", newtonx, xs, ys)
end


function main()
    args = parse_cmd()
    xs, ys = read_sample(args["datafile"])
    n = args["n"]
    λ = args["lambda"]
    fitboth(xs, ys, n; λ = λ)
end

main()
