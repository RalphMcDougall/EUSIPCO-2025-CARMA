module ModelConversions

export ARMA, CARMA, conjugate, show, step_model, BL_transformation, model_approx

using LinearAlgebra, Polynomials, FastGaussQuadrature
using Base: show

# Evaluate the integral of f on [a, b]
function integrate(f::Function, a::T, b::T) where {T<:Number}
    # Somewhat arbitrarily decided on a 10th order quadrature scheme.
    x, w = gausslegendre(10)
    x = 0.5 * (b - a) .* x .+ 0.5 * (a + b)
    w *= (b - a) / 2

    return sum(wi * f(xi) for (wi, xi) in zip(w, x))
end

# Model type definitions

abstract type AbstractModel{T<:AbstractFloat} end

struct ARMA{T} <: AbstractModel{T}
    poles::Vector{Complex{T}}
    zeros::Vector{Complex{T}}
    var::T

    function ARMA(poles::Vector{Complex{T}}, zeros::Vector{Complex{T}}, var::T) where {T}
        return if length(zeros) >= length(poles)
            error("Process must have more poles than zeros to be causal.")
        else
            new{T}(poles, zeros, var)
        end
    end
end

struct CARMA{T} <: AbstractModel{T}
    poles::Vector{Complex{T}}
    zeros::Vector{Complex{T}}
    var::T

    function CARMA(poles::Vector{Complex{T}}, zeros::Vector{Complex{T}}, var::T) where {T}
        return if length(zeros) >= length(poles)
            error("Process must have more poles than zeros to be causal.")
        else
            new{T}(poles, zeros, var)
        end
    end
end

function CARMA{T}(poles::Vector{T}, zeros::Vector{T}, var::T) where {T}
    return CARMA(Vector{Complex{T}}(poles), Vector{Complex{T}}(zeros), var)
end
function CARMA{T}(poles::Vector{Complex{T}}, zeros::Vector{T}, var::T) where {T}
    return CARMA(poles, Vector{Complex{T}}(zeros), var)
end
function CARMA{T}(poles::Vector{T}, zeros::Vector{Complex{T}}, var::T) where {T}
    return CARMA(Vector{Complex{T}}(poles), zeros, var)
end
function CARMA{T}(poles::Vector{T2}, zeros::Vector{Any}, var::T) where {T,T2}
    return if length(zeros) > 0
        error("Invalid zero vector: $(zeros)")
    else
        CARMA(poles, Vector{T2}(), var)
    end
end

function CARMA(poles::Vector{T1}, zeros::Vector{T2}, var::T3) where {T1,T2,T3}
    return CARMA{T3}(poles, zeros, var)
end

function ARMA{T}(poles::Vector{T}, zeros::Vector{T}, var::T) where {T}
    return ARMA(Vector{Complex{T}}(poles), Vector{Complex{T}}(zeros), var)
end
function ARMA{T}(poles::Vector{Complex{T}}, zeros::Vector{T}, var::T) where {T}
    return ARMA(poles, Vector{Complex{T}}(zeros), var)
end
function ARMA{T}(poles::Vector{T}, zeros::Vector{Complex{T}}, var::T) where {T}
    return ARMA(Vector{Complex{T}}(poles), zeros, var)
end
function ARMA{T}(poles::Vector{T2}, zeros::Vector{Any}, var::T) where {T,T2}
    return if length(zeros) > 0
        error("Invalid zero vector: $(zeros)")
    else
        ARMA(poles, Vector{T2}(), var)
    end
end

function ARMA(poles::Vector{T1}, zeros::Vector{T2}, var::T3) where {T1,T2,T3}
    return ARMA{T3}(poles, zeros, var)
end

# Model helper function definitions

ord(m::MT) where {T,MT<:AbstractModel{T}} = length(m.poles)
name(m::CARMA{T}) where {T} = "CARMA"
name(m::ARMA{T}) where {T} = "ARMA"

function model_approx(m1::MT, m2::MT; rtol=1E-1) where {T,MT<:AbstractModel{T}}
    #if typeof(m1) != typeof(m2)
    #    error("Model types must be the same. Provided: $(typeof(m1)) and $(typeof(m2)).")
    #end

    a1, a2 = a_vec(m1), a_vec(m2)
    b1, b2 = b_vec(m1), b_vec(m2)

    return isapprox(a1, a2; rtol=rtol) &&
           isapprox(b1, b2; rtol=rtol) &&
           isapprox(m1.var, m2.var; rtol=rtol)
end

a_vec(m::MT) where {T,MT<:AbstractModel{T}} = -fromroots(m.poles)[(ord(m) - 1):-1:0]

function b_vec(m::ARMA{T}) where {T}
    p = ord(m)
    q = length(m.zeros)
    if p > 1
        zs::Vector{Complex{T}} = zeros(p - 1)
        zs[1:q] = m.zeros

        b_poly = fromroots(zs)
        b = zeros((1, p))
        b[:] = b_poly[end:-1:0]
    else
        b = [1]
    end
    return b
end

function b_vec(m::CARMA)
    q = length(m.zeros)

    b = zeros((1, ord(m)))
    b_poly = q > 0 ? fromroots(m.zeros) : Polynomial(1)
    b[(end - q):end] = b_poly[q:-1:0]

    return b
end

function A_mat(m::MT) where {T,MT<:AbstractModel{T}}
    p = ord(m)
    A = zeros((p, p))

    A[1, :] = a_vec(m)
    for i in 2:p
        A[i, i - 1] = 1
    end

    return A
end

function Base.show(io::IO, model::MT) where {T,MT<:AbstractModel{T}}
    return print(
        io,
        "$(name(model)) model:\n * poles = $(model.poles),\n * a = $(a_vec(model)),\n * zeros = $(model.zeros),\n * b = $(b_vec(model)),\n * var = $(model.var)\n",
    )
end

# Model conjugation functions

function conjugate_pole(::Type{ARMA{T}}, lambda::Complex{T}, sample_time::T) where {T}
    return log(lambda) / sample_time
end
function conjugate_pole(::Type{CARMA{T}}, lambda::Complex{T}, sample_time::T) where {T}
    return exp(lambda * sample_time)
end

conjugate(::Type{ARMA{T}}) where {T} = CARMA{T}
conjugate(::Type{CARMA{T}}) where {T} = ARMA{T}

Theta(m::ARMA{T}, k::Integer, sample_time::T) where {T} = A_mat(m)^k
function Theta(m::CARMA{T}, k::Integer, sample_time::T) where {T}
    return exp(A_mat(m) * k * sample_time)
end

function K(m::ARMA{T}, k::Int, sample_time::T) where {T}
    p = ord(m)
    A = A_mat(m)

    Q = zeros(size(A))
    Q[1, 1] = 1.0

    # Iteratively calculate the K value for the ARMA model
    K_res = zeros(size(A))
    A_iter = Matrix(I, p, p)
    for _ in 1:k
        K_res += A_iter * Q * A_iter'
        A_iter *= A
    end

    return K_res * m.var
end

function K(m::CARMA{T}, k::Int, sample_time::T) where {T}
    A = A_mat(m)

    Q = zeros(size(A))
    Q[1, 1] = 1.0

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

    return K_res * m.var
end

function newton_raphson(
    E::Function, G::Function, H::Function, initial_state::Array{T}, E_min::T
) where {T}
    state = initial_state

    num_iters = 0
    MAX_ITERS = 5_000
    while E(state) > E_min
        state -= inv(H(state)) * G(state)

        num_iters += 1
        if num_iters == MAX_ITERS
            error(
                "Unable to converge within $(MAX_ITERS) iterations. At state $(state) with loss: $(E(state))",
            )
        end
    end
    return state, E(state), num_iters
end

EPSILON = 1E-12
first_non_zero(v::Vector) = v[findfirst(x -> abs(x) > EPSILON, v)]

function conjugate(
    model::MT, sample_time::T, quiet::Bool=false
) where {T,MT<:AbstractModel{T}}
    MT_star = conjugate(MT)

    p = ord(model)

    b = b_vec(model)

    new_poles = [conjugate_pole(MT, lambda, sample_time) for lambda in model.poles]

    index_set = 1:p

    M =
        [K(MT_star(new_poles, [], model.var), i, sample_time) for i in index_set] / model.var
    r = [only(b * K(model, i, sample_time) * b') for i in index_set] / model.var
    E_min = 1E-12

    E(v::Vector) = sum([(only(v' * Mi * v) - ri)^2 for (Mi, ri) in zip(M, r)])
    G(v::Vector) = 4 * sum([(only(v' * Mi * v) - ri) * Mi * v for (Mi, ri) in zip(M, r)])
    function H(v::Vector)
        return 4 * sum([
            (only(v' * Mi * v) - ri) * Mi + 2 * Mi * v * v' * Mi' for (Mi, ri) in zip(M, r)
        ])
    end

    initial_vec = 100 * ones(p)
    v, loss, num_iters = newton_raphson(E, G, H, initial_vec, E_min)
    if !quiet
        println("Converged to $(v) with loss $(loss) in $(num_iters) steps")
    end
    s = first_non_zero(v)
    b_star = v / s
    new_zeros = roots(Polynomial(b_star[end:-1:1]))

    new_var = model.var * s^2

    return MT_star(new_poles, new_zeros, new_var)
end

function conjugate(
    model::MT, sample_time::T, m::Vector{T}, P::Matrix{T}, quiet::Bool=false
) where {T,MT<:AbstractModel{T}}
    MT_star = conjugate(MT)

    p = ord(model)

    a = a_vec(model)
    b = b_vec(model)

    model_star = conjugate(model, sample_time, quiet)

    a_star = a_vec(model_star)
    b_star = b_vec(model_star)

    temp_mat::Matrix{T} = mapreduce(
        permutedims,
        vcat,
        [vec(b_star * Theta(model_star, i, sample_time)) for i in 0:(p - 1)],
    )
    temp_vec::Vector{T} = vec(
        mapreduce(
            permutedims, vcat, [b * Theta(model, i, sample_time) * m for i in 0:(p - 1)]
        ),
    )
    m_star::Vector{T} = inv(temp_mat) * temp_vec

    v = [transpose(b_star * Theta(model_star, i, sample_time)) for i in 0:(p^2 - 1)]
    r = [
        only(b * Theta(model, i, sample_time) * P * Theta(model, i, sample_time)' * b') for
        i in 0:(length(v) - 1)
    ]

    E_min = 1E-12

    E(M::Matrix) = sum([(only(v[i]' * M * v[i]) - r[i])^2 for i in 1:length(v)])
    function G(M::Matrix)
        return 2 *
               sum([(only(v[i]' * M * v[i]) - r[i]) * v[i] * v[i]' for i in 1:length(v)])
    end
    H(M::Matrix) = 2 * sum([(v[i] * v[i]') * (v[i] * v[i]') for i in 1:length(v)])

    temp_P_star, loss, num_iters = newton_raphson(E, G, H, zeros(size(P)), E_min)

    if !quiet
        println("Converged to $(temp_P_star) with loss $(loss) in $(num_iters) steps")
    end

    P_star::Matrix{T} = 0.5 * (temp_P_star + temp_P_star')
    if !quiet
        println("New solution $(P_star) with loss $(E(P_star))")
    end

    return m_star, P_star
end

# Forward prediction function

function step_model(
    model::MT, mean::Vector{T}, cov::Matrix{T}, steps::Int, sample_time::T
) where {T,MT<:AbstractModel{T}}
    if steps < 0
        error("Steps must be positive. Provided: $(steps)")
    end

    b = b_vec(model)

    output_means = Vector{T}()
    output_covs = Vector{T}()

    for k in 0:steps
        Th = Theta(model, k, sample_time)
        new_mean = b * Th * mean
        new_cov = b * (Th * cov * Th' + K(model, k, sample_time)) * b'

        append!(output_means, new_mean)
        append!(output_covs, new_cov)
    end
    return output_means, output_covs
end

# Implementations of the Brockwell & Linder, 2019 algorithm

function BL_transformation(model::ARMA{T}, sample_time::T) where {T}
    p = ord(model)
    q = length(model.zeros)

    new_poles = [
        conjugate_pole(typeof(model), lambda, sample_time) for lambda in model.poles
    ]

    theta_poly =
        q > 0 ? prod([Polynomial([1.0, -eta]) for eta in model.zeros]) : Polynomial([1.0])
    phi_poly = prod([Polynomial([1.0, -zeta + 0.0im]) for zeta in model.poles])

    G = [
        -zeta * (theta_poly(zeta) * theta_poly(1 / zeta)) /
        (phi_poly(zeta) * derivative(phi_poly)(1 / zeta)) for zeta in model.poles
    ]

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
    if !isapprox(imag(new_var), zero(T))
        error("Calculated variance is complex: $(new_var)")
    end
    return CARMA(new_poles, new_zeros, real(new_var))
end

function BL_transformation(model::CARMA{T}, sample_time::T) where {T}
    p = ord(model)
    q = length(model.zeros)
    new_poles = [
        conjugate_pole(typeof(model), lambda, sample_time) for lambda in model.poles
    ]
    b = b_vec(model)

    b_poly = q > 0 ? fromroots(model.zeros) : Polynomial(1)
    a_poly = fromroots(model.poles)

    K = [
        (b_poly(lambda) * b_poly(-lambda)) / (a_poly(-lambda) * derivative(a_poly)(lambda))
        for lambda in model.poles
    ]

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

    new_zeros = Vector{Complex{T}}()
    for z in k_zeros
        eta1, eta2 = z + sqrt(z^2 - 1), z - sqrt(z^2 - 1)
        if abs(eta1) < 1
            append!(new_zeros, [eta1])
        else
            append!(new_zeros, [eta2])
        end
    end

    new_var = model.var * prod(new_poles) / prod(new_zeros) * (-2)^(p - r) * cr
    if !isapprox(imag(new_var), zero(T))
        error("Calculated variance is complex: $(new_var)")
    end
    return ARMA(new_poles, new_zeros, real(new_var))
end

end # module