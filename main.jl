using Revise

using Plots, LaTeXStrings, Distributions, LinearAlgebra, ProgressMeter, Random

includet("ModelConversions.jl")
using .ModelConversion


BLACK   = colorant"#000000"
BLUE    = colorant"#030aa7" # XKCD: "Cobalt blue"
RED     = colorant"#9a0200" # XKCD: "Deep red"
GREEN   = colorant"#02590f" # XKCD: "Deep green"
GREY    = colorant"#929591" # XKCD: "Grey"

LINE_WIDTH = 2
SECONDARY_LINEWIDTH = 1
LEGEND_FONT_SIZE = 10

SCATTER_SIZE = 2
GAUSSIAN_CONFIDENCE_95 = 1.96


function stationary_process_tracking()
    println("\nRunning stationary process tracking example...\n")
    sample_time = 0.5

    arma = ARMA([0.98 * exp(im * 0.25), 0.98 * exp(-im * 0.25)], [], 5E-1)
    println(arma)
    carma = conjugate(arma, sample_time)
    println(carma)
    bl2019_carma = BL_transformation(arma, sample_time)
    println("BL2019 $(bl2019_carma)")


    initial_m::Vector{AbstractFloat} = [3.0; -1.0]
    initial_P::Matrix{AbstractFloat} = [2.0 0.5; 0.5 1.0]

    initial_m_star, initial_P_star = conjugate(arma, sample_time, initial_m, initial_P)
    println("Transformed mean: $(initial_m_star)")
    println("Transformed cov: $(initial_P_star)") 

    num_steps = 50

    num_sample_paths = 1_000
    sample_paths::Vector{Vector{AbstractFloat}} = []
    for _ in 1:num_sample_paths
        state = initial_m + Matrix(cholesky(initial_P)) * vec(rand(Normal(0, 1), 2))
        b = ModelConversion.b_vec(arma)
        A = ModelConversion.A_mat(arma)

        path::Vector{AbstractFloat} = b * state
        for _ in 1:num_steps
            state = A * state 
            state[1] += rand(Normal(0, sqrt(arma.var)))
            append!(path, b * state)
        end
        append!(sample_paths, [path])
    end



    arma_steps = 0:sample_time:(sample_time * num_steps)
    arma_means, arma_covs = step_model(arma, initial_m, initial_P, num_steps, sample_time)

    arma_upper_conf, arma_lower_conf = arma_means + GAUSSIAN_CONFIDENCE_95 * sqrt.(arma_covs), arma_means - GAUSSIAN_CONFIDENCE_95 * sqrt.(arma_covs)

    upsampling = 10
    carma_steps = 0:(sample_time / upsampling):(sample_time * num_steps)
    carma_means, carma_covs = step_model(carma, initial_m_star, initial_P_star, num_steps * upsampling, sample_time / upsampling)
    carma_upper_conf, carma_lower_conf = carma_means + GAUSSIAN_CONFIDENCE_95 * sqrt.(carma_covs), carma_means - GAUSSIAN_CONFIDENCE_95 * sqrt.(carma_covs)
    
    p = plot(grid=false, xlabel="Forward prediction (s)", ylabel="Process estimate", ylim=[-30, 30], title="Process prediction")
    
    plot!(carma_steps, carma_means, color=BLUE, label="", linewidth=LINE_WIDTH)
    plot!(carma_steps, carma_upper_conf, color=GREY, label="", linewidth=SECONDARY_LINEWIDTH, linestyle=:dash)
    plot!(carma_steps, carma_lower_conf, color=GREY, label="", linewidth=SECONDARY_LINEWIDTH, linestyle=:dash)
    
    scatter!(arma_steps, arma_means, color=BLACK, label="", markersize=SCATTER_SIZE)
    scatter!(arma_steps, arma_upper_conf, color=BLACK, label="", markersize=SCATTER_SIZE)
    scatter!(arma_steps, arma_lower_conf, color=BLACK, label="", markersize=SCATTER_SIZE)

    hline!([0], linewidth=1, color=BLACK, label="")

    scatter!(1:0,1:0, color=BLACK, label="ARMA", markersize=SCATTER_SIZE)
    plot!(1:0, 1:0, color=BLUE, label="CARMA", linewidth=LINE_WIDTH)
    plot!(1:0, 1:0, color=BLUE, label="95% error", linewidth=SECONDARY_LINEWIDTH, linestyle=:dash)


    display(p)
    savefig(p, "figs/stationary_process.pdf")
    println("Finished stationary process prediction example.\n")

end


function transform_integrated_OU()
    sample_time = 1.0
    alpha_range = -10:0.001:0
    discrete_zeros::Vector{Real} = []
    discrete_vars::Vector{Real} = []
    last = -1
    for alpha in alpha_range
        carma = CARMA([0, alpha], [], 1.0)
        arma = conjugate(carma, sample_time)
        append!(discrete_zeros, real.(arma.zeros))
        append!(discrete_vars, [arma.var])
        if abs(discrete_zeros[end] - last) > 1E-2
            println(alpha)
        end
        last = discrete_zeros[end]
    end

    p = plot(alpha_range, discrete_vars, yaxis=:log, label=L"\sigma_d^2", xlabel=L"\alpha", ylabel="Variance", legend=:bottomleft, legendfontsize=LEGEND_FONT_SIZE, color=RED, linewidth=LINE_WIDTH, ylim=[10^(-2.5), 1E0], grid=false)
    plot!(twinx(p), alpha_range, discrete_zeros, title="Non-trivial ARMA(2, 1) parameters", label=L"z_d", legend=:bottomright, legendfontsize=LEGEND_FONT_SIZE, ylabel="Zero position", color=BLUE, linewidth=LINE_WIDTH, ylim=[-1.0, 0.0], grid=false)
    
    display(p)
    savefig(p, "figs/ou_transformation.pdf")

end


function transform_singer()
    sample_time = 1.0
    alpha_range = -10:0.001:0
    discrete_zeros::Matrix{Complex} = zeros((2, length(alpha_range)))
    discrete_vars::Vector{Real} = []
    for (ind, alpha) in enumerate(alpha_range)
        carma = CARMA([0, 0, alpha], [], 1.0)
        arma = conjugate(carma, sample_time)
        discrete_zeros[:, ind] = arma.zeros
        append!(discrete_vars, [arma.var])
    end
    
    p = plot(alpha_range, discrete_vars, yaxis=:log, label=L"\sigma_d^2", xlabel=L"\alpha", legend=:bottomleft, legendfontsize=LEGEND_FONT_SIZE, ylabel="Variance", color=RED, linewidth=LINE_WIDTH, grid=false, ylim=[1E-3, 1E-1])
    
    plot!(twinx(p), alpha_range, real.(discrete_zeros[1,:]), title="Non-trivial ARMA(3,2) parameters", label=L"z_d", legend=:bottomright, legendfontsize=LEGEND_FONT_SIZE, ylabel="Zero position", color=BLUE, linewidth=LINE_WIDTH, grid=false, ylim=[-2.5, 0])
    plot!(twinx(p), alpha_range, real.(discrete_zeros[2,:]), label="", color=BLUE, linewidth=LINE_WIDTH, ylim=[-2.5, 0.0])

    display(p)
    savefig(p, "figs/singer_transformation.pdf")
end


function run_benchmarks()
    println("Benchmark:\n-------------------------\n")

    println("Running BL2019 example")
    # This is the example from Brockwell & Lindner, 2019
    carma = CARMA([-0.5, -1.0], [-0.25], 1.0)
    sample_time = 0.1
    println(carma)

    target_bl2019_arma = ARMA([exp(-0.05), exp(-0.1)], [0.9752813889], 0.08842703)

    bl2019_arma = BL_transformation(carma, sample_time)
    if !model_approx(bl2019_arma, target_bl2019_arma)
        error("BL transformation target!\n\n$(bl2019_arma)\n\n$(target_bl2019_arma)")
    end
    bl2019_carma = BL_transformation(bl2019_arma, sample_time)
    if !model_approx(bl2019_carma, carma)
        error("BL transformation does not match original model!\n\n$(bl2019_carma)\n\n$(carma)")
    end


    arma = conjugate(carma, sample_time)
    println(arma)
    if !model_approx(arma, target_bl2019_arma)
        error("BL2019 ARMA model does not match calculated conjugate!")
    end
    conjugate_carma = conjugate(arma, sample_time)
    println(conjugate_carma)
    if !model_approx(conjugate_carma, carma)
        error("Does not match original!")
    end
    println("\nPassed BL2019 example.\n")

    # There is definitely numerical error present.

    num_tests = 10
    prog = Progress(num_tests, "Evaluating random models...")
    for _ in 1:num_tests
        sample_time = 10^rand(Uniform(-1, 1))
        var = 10^rand(Uniform(-2, 0))

        poles = [rand(Uniform(-3, 0)) for _ in 1:2]
        zeros = [rand(Uniform(-3, 0)) for _ in 1:(length(poles) - 1)]

        carma = CARMA(poles, zeros, var)
        bl2019_conj_arma = BL_transformation(carma, sample_time)
        bl2019_double_conj = BL_transformation(bl2019_conj_arma, sample_time)
        if !model_approx(carma, bl2019_double_conj)
            error("BL: Found model that doesn't return to itself after conjugation.\n\n$(carma)\n\n$(bl2019_conj_arma)\n\n$(bl2019_double_conj)")
        end
        
        conj_arma = conjugate(carma, sample_time)
        double_conj = conjugate(conj_arma, sample_time)
        #println("CHECKING $(carma)")

        if !model_approx(conj_arma, bl2019_conj_arma)
            error("BL and our method provide different conjugates:\n\nBL: $(bl2019_conj_arma)\n\n $(conj_arma)")
        end

        if !model_approx(carma, double_conj)
            error("Found model that doesn't return to itself after conjugation.\n\n$(carma)\n\n$(conj_arma)\n\n$(double_conj)")
        end
        next!(prog)
    end


    println("-------------------------\n")
end


function main()
    Random.seed!(1234)
    run_benchmarks()

    # Stationary example
    stationary_process_tracking()
    
    # Non-stationary examples
    transform_integrated_OU()
    transform_singer()
end