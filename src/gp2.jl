function gp2(x₀, y; iterations = 1)

    # reshape so that this works with KernelFunctions.ColVecs
    x = reshape(x₀,1, length(x₀))

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

    helper(p) = negativemarginallikelihood_gp2(p, K, x, y, zeromean)

    # use DifferentiationInterface to get gradients

    backend = AutoMooncake(config = nothing)
    
    prep = prepare_gradient(negativemarginallikelihood_gp2, backend, initialsol, Cache(K), Constant(x), Constant(y), Constant(zeromean))   

    gradhelper!(grad, p) = DifferentiationInterface.gradient!(negativemarginallikelihood_gp2, grad, prep, backend, p, Cache(K), Constant(x), Constant(y), Constant(zeromean))

    optimize(helper, gradhelper!, initialsol, ConjugateGradient(), opt).minimizer

end

# Negative marginal likelihood function of gp
function negativemarginallikelihood_gp2(p, K, x, y, zeromean)

    N = length(y)

    θ = exp.(p) # make parameters positive

    # Calculate covariance matrix
    kernelmatrix!(K, θ[1] * with_lengthscale(SEKernel(), θ[2]), ColVecs(x))

    # add jitter on diagonal
    for n in 1:N
        K[n, n] += 1e-6
    end

    
    # Two different ways of calculating log marginal likelihood

    # 1. Using Distributions.jl
    # logl = logpdf(MvNormal(zeromean, K + θ[3]*I), y)

    # 2. Using Cholesky decomposition
    logl = let 

        C = cholesky(K + θ[3]*I).L
        
        invCy = C\y

        -sum(log, diag(C)) - 0.5*sum(abs2.(invCy)) - 0.5*N*log(2π)
        
    end

    # Return negative log marginal likelihood, we want to minimise this.
    return -logl

end