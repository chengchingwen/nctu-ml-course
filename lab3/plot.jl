function plot_3line((f, f2, f3))
    lb, ub = (-2, 2)
    plt = plot(x->f(x), lb, ub, color=:black)
    plot!(plt, x->f2(x), lb, ub, color=:red)
    plot!(plt, x->f3(x), lb, ub, color=:red)
    plt
end

function plot_result((t, p, p10, p50), xs, ys)
    pt = plot_3line(t)
    title!(pt, "Ground truth")

    pp = plot_3line(p)
    title!(pp, "Predict result")
    scatter!(pp, xs[1], ys[1], color=:cyan)

    pp10 = plot_3line(p10)
    title!(pp10, "After 10 incomes")
    scatter!(pp10, xs[2], ys[2], color=:cyan)

    pp50 = plot_3line(p50)
    title!(pp50, "After 50 incomes")
    scatter!(pp50, xs[3], ys[3], color=:cyan)

    plot(pt, pp, pp10, pp50, layout=(2,2), legend=false)
end
