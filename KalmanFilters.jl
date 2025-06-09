module KalmanFilter

export KalmanFilter, update, get_output

using .ModelConversion

ssquare(M::Array{Float64}) = (ndims(M) == 2) && (size(M, 1) == size(M, 2))
canmultiply(A::Array{Float64}, B::Array{Float64}) = (size(A, 2) == size(B, 1))

copy(kf::KalmanFilter) = KalmanFilter(kf.A, kf.Q, kf.H, kf.R, kf.m, kf.P)


struct KalmanFilter
    A::Matrix{Float64}
    Q::Matrix{Float64}
    H::Matrix{Float64}
    R::Matrix{Float64}
    m::Vector{Float64}
    P::Matrix{Float64}

    
    function KalmanFilter(A::Matrix{Float64}, Q::Matrix{Float64}, H::Matrix{Float64}, R::Matrix{Float64}, m::Vector{Float64}, P::Matrix{Float64})
        if !issquare(A)
            error("The provided A matrix is not square. Provided: " * string(size(A)))
        end
        if !canmultiply(A, m)
            error("The provided A and m matrices cannot be multiplied. Provided sizes: " * string(size(A)) * " and " * string(size(m)))
        end
        if !issquare(Q)
            error("The provided Q matrix is not square. Provided: " * string(size(Q)))
        end
        
        if !canmultiply(H, m)
            error("The provided H and m matrices cannot be multiplied. Provided sizes: " * string(size(H)) * " and " * string(size(m)))
        end
        if !issquare(R)
            error("The provided R matrix is not square. Provided: " * string(size(R)))
        end
        if size(R, 2) != size(H, 1)
            error("The provided H and R matrices are not compatible. Provided sizes: " * string(size(H)) * " and " * string(size(R)))
        end

        if !issquare(P)
            error("The provided P matrix is not square. Provided: " * string(size(P)))
        end
        if size(P, 2) != size(m, 1)
            error("The provided m and P matrices are not compatible. Provided sizes: " * string(size(m)) * " and " * string(size(P)))
        end

        new(A, Q, H, R, m, P)
    end
end

function KalmanFilter(model::ARMA, mean::Vector{Float64}, cov::Matrix{Float64}, sensor_noise::Float64)
    p = ord(model)
    
    A = A_mat(model)
    Q = zeros((p, p))
    Q[1,1] = model.var
    H = b_vec(model)
    R = ones((1,1)) * sensor_noise

    KalmanFilter(A, Q, H, R, mean, cov)
end

function predict_state(filter::KalmanFilter, steps::Integer = 1)
    if steps < 0
        error("Unable to use negative number of steps: " * string(steps))
    end

    m_pred, P_pred = filter.m, filter.P

    A, Q = filter.A, filter.Q

    for _ in 1:steps
        m_pred = A * m_pred
        P_pred = A * P_pred * A' + Q 
    end

    m_pred, P_pred
end

function get_output(filter::KalmanFilter)
    H, R = filter.H, filter.R

    H * filter.m, H * filter.P * H' + R
end


function update(filter::KalmanFilter, observation::Vector{Float64})
    m_pred, P_pred = predict_state(filter)
    
    A, Q, H, R, _, _ = filter.A, filter.Q, filter.H, filter.R, filter.m, filter.P
    if length(observation) != 0 
        innovation = observation - H * m_pred
        
        S = H * P_pred * H' + R
        K = P_pred * H' * inv(S)
        new_m = m_pred + K * innovation
        new_P = P_pred - K * S * K'
    else
        new_m = m_pred
        new_P = P_pred
    end
    KalmanFilter(A, Q, H, R, new_m, new_P)
end
update(filter::KalmanFilter) = update(filter, convert(Vector{Float64}, []))


function predict_output(filter::KalmanFilter, steps::Integer = 1)
    m_pred, P_pred = predict_state(filter, steps)
    H, R = filter.H, filter.R
    H * m_pred, H * P_pred * H' + R
end


function loglikelihood(filter::KalmanFilter, data::Vector{Vector{Float64}})
    res = 0

    N = length(data)
    for i in 1:N
        pred_mean, pred_cov = predict_output(filter)
        dist = Distributions.MultivariateNormal(vec(pred_mean), pred_cov)
        res += Distributions.loglikelihood(dist, data[i])

        filter = update(filter, data[i])
    end

    res
end


end # module