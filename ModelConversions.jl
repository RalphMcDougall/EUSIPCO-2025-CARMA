module ModelConversion

export ARMA, CARMA, conjugate, show, step_model, BL_transformation, model_approx

using LinearAlgebra, Polynomials, FastGaussQuadrature
using Base: show


# Evaluate the integral of f on [a, b]
function integrate(f::Function, a::Real, b::Real)
    # Somewhat arbitrarily decided on a 10th order quadrature scheme.
    x, w = gausslegendre(10)
    x = 0.5 * (b - a) .* x .+ 0.5 * (a + b)
    w *= (b - a) / 2

    sum(wi * f(xi) for (wi, xi) in zip(w, x))
end


# Model type definitions

abstract type Model end

struct ARMA <: Model
    poles::Vector{Complex{AbstractFloat}}
    zeros::Vector{Complex{AbstractFloat}}
    var::AbstractFloat
end

struct CARMA <: Model
    poles::Vector{Complex{AbstractFloat}}
    zeros::Vector{Complex{AbstractFloat}}
    var::AbstractFloat
end


# Model helper function definitions

ord(m::Model) = length(m.poles)

function model_approx(m1::Model, m2::Model; rtol=1E-1)
    if typeof(m1) != typeof(m2)
        error("Model types must be the same. Provided: $(typeof(m1)) and $(typeof(m2)).")
    end

    a1, a2 = a_vec(m1), a_vec(m2)
    b1, b2 = b_vec(m1), b_vec(m2)

    return isapprox(a1, a2, rtol=rtol) && isapprox(b1, b2, rtol=rtol) && isapprox(m1.var, m2.var, rtol=rtol)
end

a_vec(m::Model) = -fromroots(m.poles)[ (ord(m) - 1):-1:0 ]

function b_vec(m::ARMA)
    p = ord(m)
    q = length(m.zeros)
    if p > 1
        zs::Vector{Complex} = zeros(p - 1)
        zs[1:q] = m.zeros

        b_poly = fromroots(zs)
        b = zeros((1, p))
        b[:] = b_poly[end:-1:0]
    else
        b = [1]
    end
    b
end

function b_vec(m::CARMA)
    q = length(m.zeros)

    b = zeros((1, ord(m)))
    b_poly = q > 0 ? fromroots(m.zeros) : Polynomial(1)
    b[end-q:end] = b_poly[q:-1:0]

    b
end

function A_mat(m::Model) 
    p = ord(m)
    A = zeros((p, p))

    A[1,:] = a_vec(m)
    for i in 2:p
        A[i, i - 1] = 1
    end

    A
end

Base.show(io::IO, model::ARMA) = print(io, "ARMA model:\n * poles = $(model.poles),\n * a = $(a_vec(model)),\n * zeros = $(model.zeros),\n * b = $(b_vec(model)),\n * var = $(model.var)\n")
Base.show(io::IO, model::CARMA) = print(io, "CARMA model:\n * poles = $(model.poles),\n * a = $(a_vec(model)),\n * zeros = $(model.zeros),\n * b = $(b_vec(model)),\n * var = $(model.var)\n")


# Model conjugation functions

conjugate_pole(T::Type{ARMA}, lambda::Complex, sample_time::AbstractFloat) = log(lambda) / sample_time
conjugate_pole(T::Type{CARMA}, lambda::Complex, sample_time::AbstractFloat) = exp(lambda * sample_time)

conjugate(T::Type{ARMA}) = CARMA
conjugate(T::Type{CARMA}) = ARMA

Theta(m::ARMA, k::Integer, sample_time::AbstractFloat) = A_mat(m)^k 
Theta(m::CARMA, k::Integer, sample_time::AbstractFloat) = exp(A_mat(m) * k * sample_time)

function K(m::ARMA, k::Integer, sample_time::AbstractFloat)
    p = ord(m)
    A = A_mat(m)

    Q = zeros(size(A))
    Q[1,1] = 1.0

    # Iteratively calculate the K value for the ARMA model
    K_res = zeros(size(A))
    A_iter = Matrix(I, p, p)
    for _ in 1:k 
        K_res += A_iter * Q * A_iter'
        A_iter *= A
    end

    K_res * m.var
end

function K(m::CARMA, k::Integer, sample_time::AbstractFloat)
    A = A_mat(m)

    Q = zeros(size(A))
    Q[1,1] = 1.0

    # Calculate the K value by integrating over all intervals [(i - 1) * T_s, i * T_s]
    # The integration is done over every interval instead of the whole interval directly 
    # to reduce numerical error
    K_foo(tau) = begin
        exp_term = exp(A * (k * sample_time - tau))
        exp_term * Q * exp_term'
    end

    K_res = zeros(size(A))
    for i in 1:k 
        K_res += integrate(K_foo, (i - 1) * sample_time, i * sample_time)
    end

    K_res * m.var
end

function newton_raphson(E::Function, G::Function, H::Function, initial_state::Array, E_min::Real)
    state = initial_state

    num_iters = 0
    MAX_ITERS = 5_000
    while E(state) > E_min
        state -= inv(H(state)) * G(state)

        num_iters += 1
        if num_iters == MAX_ITERS
            error("Unable to converge within $(MAX_ITERS) iterations. At state $(state) with loss: $(E(state))")
        end
    end
    state, E(state), num_iters
end

EPSILON = 1E-12
first_non_zero(v::Vector) = v[findfirst(x -> abs(x) > EPSILON, v)]

function conjugate(model::Model, sample_time::AbstractFloat, quiet::Bool=false)
    T = typeof(model)
    T_star = conjugate(T)

    p = ord(model)

    b = b_vec(model)

    new_poles = [conjugate_pole(T, lambda, sample_time) for lambda in model.poles]

    index_set = 1:p

    M = [K(T_star(new_poles, [], model.var), i, sample_time) for i in index_set] / model.var
    r = [only(b * K(model, i, sample_time) * b') for i in index_set] / model.var
    E_min = 1E-12

    E(v::Vector) = sum([(only(v' * Mi * v) - ri)^2 for (Mi, ri) in zip(M, r)])
    G(v::Vector) = 4 * sum([(only(v' * Mi* v) - ri) * Mi * v  for (Mi, ri) in zip(M, r)])
    H(v::Vector) = 4 * sum([(only(v' * Mi * v) - ri) * Mi + 2 * Mi * v * v' * Mi' for (Mi, ri) in zip(M, r)])

    initial_vec = 100 * ones(p)
    v, loss, num_iters = newton_raphson(E, G, H, initial_vec, E_min)
    if !quiet
        println("Converged to $(v) with loss $(loss) in $(num_iters) steps")
    end
    s = first_non_zero(v)
    b_star = v / s 
    new_zeros = roots(Polynomial(b_star[end:-1:1]))

    new_var = model.var * s^2

    T_star(new_poles, new_zeros, new_var)
end

function conjugate(model::Model, sample_time::AbstractFloat, m::Vector, P::Matrix, quiet::Bool=false)
    T = typeof(model)
    T_star = conjugate(T)

    p = ord(model)

    a = a_vec(model)
    b = b_vec(model)

    model_star = conjugate(model, sample_time, quiet)

    a_star = a_vec(model_star)
    b_star = b_vec(model_star)

    temp_mat::Matrix{AbstractFloat} = mapreduce(permutedims, vcat, [vec(b_star * Theta(model_star, i, sample_time)) for i in 0:(p - 1)])
    temp_vec::Vector{AbstractFloat} = vec(mapreduce(permutedims, vcat, [b * Theta(model, i, sample_time) * m for i in 0:(p - 1)]))
    m_star::Vector{AbstractFloat} = inv(temp_mat) * temp_vec
    
    v = [transpose(b_star * Theta(model_star, i, sample_time)) for i in 0:(p^2 - 1)]
    r = [only(b * Theta(model, i, sample_time) * P * Theta(model, i, sample_time)' * b') for i in 0:(length(v) - 1)]

    E_min = 1E-12

    E(M::Matrix) = sum([(only(v[i]' * M * v[i]) - r[i])^2 for i in 1:length(v)])
    G(M::Matrix) = 2 * sum([(only(v[i]' * M * v[i]) - r[i]) * v[i] * v[i]'  for i in 1:length(v)])
    H(M::Matrix) = 2 * sum([(v[i] * v[i]') * (v[i] * v[i]') for i in 1:length(v)])

    temp_P_star, loss, num_iters = newton_raphson(E, G, H, zeros(size(P)), E_min)

    if !quiet
        println("Converged to $(temp_P_star) with loss $(loss) in $(num_iters) steps")
    end

    P_star::Matrix{AbstractFloat} = 0.5 * (temp_P_star + temp_P_star')
    if !quiet
        println("New solution $(P_star) with loss $(E(P_star))")
    end

    m_star, P_star
end


# Forward prediction function

function step_model(model::Model, mean::Vector{AbstractFloat}, cov::Matrix{AbstractFloat}, steps::Integer, sample_time::AbstractFloat)
    if steps < 0
        error("Steps must be positive. Provided: $(steps)")
    end

    b = b_vec(model)

    output_means::Vector{AbstractFloat} = []
    output_covs::Vector{AbstractFloat} = []

    for k in 0:steps
        T = Theta(model, k, sample_time)
        new_mean = b * T * mean
        new_cov = b * (T * cov * T' + K(model, k, sample_time)) * b' 

        append!(output_means, new_mean)
        append!(output_covs, new_cov)
    end
    output_means, output_covs
end


# Implementations of the Brockwell & Linder, 2019 algorithm

function BL_transformation(model::ARMA, sample_time::AbstractFloat)
    p = ord(model)
    q = length(model.zeros)

    new_poles = [conjugate_pole(typeof(model), lambda, sample_time) for lambda in model.poles]

    theta_poly = q > 0 ? prod([Polynomial([1.0, -eta]) for eta in model.zeros]) : Polynomial([1.0])
    phi_poly = prod([Polynomial([1.0, -zeta + 0.0im]) for zeta in model.poles])

    G = [-zeta * (theta_poly(zeta) * theta_poly(1 / zeta)) / (phi_poly(zeta) * derivative(phi_poly)(1 / zeta)) for zeta in model.poles]

    g_poly = Polynomial(0)
    for j in 1:p
        prod = Polynomial(1)
        for m in 1:p 
            if j == m
                continue
            end
            prod *= Polynomial([new_poles[m]^2, 1])
        end
        g_poly += prod * G[j] * new_poles[j]
    end
    S = g_poly[end]
    g_zeros = roots(g_poly)
    new_zeros = -sqrt.(-g_zeros)

    new_var = -2 * S * model.var
    CARMA(new_poles, new_zeros, new_var)
end

function BL_transformation(model::CARMA, sample_time::AbstractFloat)
    p = ord(model)
    q = length(model.zeros)
    new_poles = [conjugate_pole(typeof(model), lambda, sample_time) for lambda in model.poles]
    b = b_vec(model)


    b_poly = q > 0 ? fromroots(model.zeros) : Polynomial(1)
    a_poly = fromroots(model.poles)

    K = [(b_poly(lambda) * b_poly(-lambda)) / (a_poly(-lambda) * derivative(a_poly)(lambda)) for lambda in model.poles]

    k_poly = Polynomial(0)
    for j in 1:p
        prod = Polynomial(1)
        for m in 1:p 
            if j == m
                continue
            end
            prod *= Polynomial([-cosh(model.poles[m] * sample_time), 1])
        end
        k_poly += prod * K[j] * sinh(model.poles[j] * sample_time)
    end
    k_zeros = roots(k_poly)
    r = length(k_zeros)
    cr = k_poly[r]

    new_zeros::Vector{Complex} = []
    for z in k_zeros
        eta1, eta2 = z + sqrt(z^2 - 1), z - sqrt(z^2 - 1)
        if abs(eta1) < 1
            append!(new_zeros, [eta1])
        else
            append!(new_zeros, [eta2])
        end
    end

    new_var = model.var * prod(new_poles) / prod(new_zeros) * (-2)^(p - r) * cr
    ARMA(new_poles, new_zeros, new_var)
end


end # module