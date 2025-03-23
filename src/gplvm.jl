"""
Y are the D×N high-dimensional data points
iterations is the number of iterations of the optimisation algorithm
Q is the dimensionality of the latent space
"""
function gplvm(Y; iterations = 1, Q = 2)

    # Get number of data items
    D, N = size(Y)

    # Allocate once zero vector necessary for marginal likelihood 
    # calculation of zero-mean Gaussian process
    zerovector = zeros(N)

    # Initialise parameters randomly:
    # first Q*N elements are the N latent Q-dimensional projections X
    # next 2 elements are kernel parameters - take log here because unpack function uses exp to ensure positivity
    # last parameter is the noise variance  - take log here because unpack function uses exp to ensure positivity
    
    rng = MersenneTwister(1234)

    initialsol = [randn(rng, Q*N)*0.1; log(1.0); log(1.0); log(1.0)]

    # setup optimiser options
    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    # use DifferentiationInterface to get gradients
    backend = AutoMooncake(config = nothing)
    
    prep = prepare_gradient(negativemarginallikelihood, backend, initialsol, Constant(Y), Constant(zerovector), Constant(D), Constant(Q), Constant(N))   

    gradhelper!(grad, p) = DifferentiationInterface.gradient!(negativemarginallikelihood, grad, prep, backend, p, Constant(Y), Constant(zerovector), Constant(D), Constant(Q), Constant(N))

    helper(p) = negativemarginallikelihood(p, Y, zerovector, D, Q, N)

    # call actual optimisation
    finalsolution = optimize(helper, gradhelper!, initialsol, ConjugateGradient(), opt).minimizer

    # obtain optimised latent
    X = unpack_gplvm(finalsolution, Q, N)[1]

    # return projections
    return X 

end



# Negative marginal likelihood function of GPLVM.
# We want to minimise this.
function negativemarginallikelihood(p, Y, zerovector, D, Q, N)

    # extract parameters from vector p
    X, θ, σ² = unpack_gplvm(p, Q, N)

    # calculate pairwise squared Euclidean distances
    SqEu = pairwise(SqEuclidean(), X)

    # calculate covariance matrix and add jitter for numerical stability 
    K = Symmetric(covariance(SqEu, θ)) + 1e-6*I

    # accummulate here log likelihood over D dimensions
    accloglikel = zero(eltype(p))

    # instiantiate multivariate normal distribution
    mvn = MvNormal(zerovector, K + σ²*I)

    # iterate over D dimensions
    for d in 1:D

        # calculate log likelihood of d-th dimension
        accloglikel += @views logpdf(mvn, Y[d, :])

    end

    # return negative log marginal likelihood
    -1.0 * accloglikel

end


# Calculates covariance matrix of Gaussian process
covariance(SqEu, θ) = Symmetric(θ[1] .* exp.(-SqEu .* θ[2]))


# Given parameters flattened in p, unpack them into X, θ and σ²
function unpack_gplvm(p, Q, N)

    MARK = 0

    # First Q*N elements are the N latent Q-dimensional projections X
    X = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

    # Next two elements are kernel parameters
    θ = exp.(p[MARK+1:MARK+2]); MARK += 2

    # The last parameter is the noise variance
    σ² = exp(p[MARK+1]); MARK += 1

    return X, θ, σ²

end