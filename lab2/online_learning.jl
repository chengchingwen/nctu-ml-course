function counts(s)
    total = 0
    one = 0
    for c ∈ s
        total +=1
        if c == '1'
            one += 1
        end
    end
    one, total
end

mutable struct BetaBinomial
    a::Int
    b::Int
    m::Int
    N::Int
    BetaBinomial(a::Int, b::Int) = new(a, b, 0, 0)
end

update!(bb::BetaBinomial, sample::String) = update!(bb, counts(sample)...)
function update!(bb::BetaBinomial, m::Int, N::Int)
    bb.a += m
    bb.b += N - m
    bb.m += m
    bb.N += N
    bb
end

function learn!(bb::BetaBinomial, samples)
    for (i, sample) ∈ enumerate(samples)
        priorp = (a=bb.a, b=bb.b)
        m, N = counts(sample)
        update!(bb, m, N)
        θ = m / N
        likelihood = binomial(N, m) * θ^m * (1 - θ)^(N-m)
        posteriorp = (a=bb.a, b=bb.b)
        println("Case $i: $sample")
        println("Likelihood: $likelihood")
        println("Beta prior: a = $(priorp.a), b = $(priorp.b)")
        println("Beta posterior: a = $(posteriorp.a), b = $(posteriorp.b)")
        println()
    end
    bb
end
