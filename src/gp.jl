function gp(x, y; iterations = 1)

    # Get number of data items
    N = length(y)

    # Allocate once zero vector necessary for marginal likelihood 
    zeromean = zeros(N)

    # Initialise parameters randomly
    rng = MersenneTwister(1234)

    initialsol = randn(rng, 3)
    
    # pre-allocate N×N covariance matrix K
    K = zeros(N, N)

    # setup optimiser options
    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    helper(p) = negativemarginallikelihood_gp(p, K, x, y, zeromean)

    # use DifferentiationInterface to get gradients

    backend = AutoMooncake(config = nothing)
    
    prep = prepare_gradient(negativemarginallikelihood_gp, backend, initialsol, Cache(K), Constant(x), Constant(y), Constant(zeromean))   

    gradhelper!(grad, p) = DifferentiationInterface.gradient!(negativemarginallikelihood_gp, grad, prep, backend, p, Cache(K), Constant(x), Constant(y), Constant(zeromean))

    optimize(helper, gradhelper!, initialsol, ConjugateGradient(), opt).minimizer

end

# Negative marginal likelihood function of gp
function negativemarginallikelihood_gp(p, K, x, y, zeromean)

    N = length(y)

    θ = exp.(p) # make parameters positive

    # Calculate covariance matrix
    for m in 1:N
        for n in 1:N
           K[n, m] =  θ[1] * exp(-0.5 * abs2(x[n] - x[m])/ θ[2])
        end
    end

    # add jitter on diagonal
    for n in 1:N
        K[n, n] += 1e-6
    end

    # Return negative log marginal likelihood.
    # We want to minimise this.
    return -logpdf(MvNormal(zeromean, K + θ[3]*I), y)

end