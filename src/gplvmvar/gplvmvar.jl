function gplvmvar(X, 𝛔 = missing; iterations = 1, η = 1e-2, seed = 1, Q = 2, JITTER = 1e-6,  H1 = 10, H2 = H1, VERIFY = false)
    
    #---------------------------------------------------------------------------
    # Setup variables and free parameters: set random seed, get dimensions, etc
    #---------------------------------------------------------------------------

    # fix random seed for reproducibility

    rg = MersenneTwister(seed)

    # get dimensions of data

    D, N = size(X)
    
    # report to user

    @printf("Running gplvmvarnet.\n")
    @printf("There are %d number of data items\n", N)
    @printf("There are %d number of dimensions\n", D)
    @printf("Q=%d\n", Q)

    # we work with precisions instead of standard deviations

    𝛃 = inverterrors(𝛔)


    # define neural network that modeltypes variational parameters and its number of weights

    net = ThreeLayerNetwork(in = Q, H1 = H1, H2 = H2, out = D)
    
    nwts = ForwardNeuralNetworks.numweights(net)
    
    
    # Initialise free parameters randomly

    p0 = let 
        
        ismissing(𝛃) ? [randn(rg, Q*N)*0.2; randn(rg,3)*1; 0.1*randn(rg, nwts); randn(rg, N); randn(rg)] :
                       [randn(rg, Q*N)*0.2; randn(rg,2)*1; 0.1*randn(rg, nwts); randn(rg, N); randn(rg)]
        
    end
    
    # define auxiliary unpack function

    upk(p, 𝛃) = unpack_gplvmvar(p, 𝛃, D, N, net, Q)


    #---------------------------------------------------------------------------
    # Define optimisation problem
    #---------------------------------------------------------------------------
    
    # setup objective function and gradient

    objective(p) = -marginallikelihood_gplvmvar(X, upk(p, 𝛃)...; JITTER = JITTER, η = η)
    
    VERIFY ? numerically_verify_gplvmplus(X, upk(p0, 𝛃)..., JITTER, η) : nothing
    
    @printf("Optimising %d number of parameters\n",length(p0))
    optf = Optimization.OptimizationFunction((u,_)->objective(u), Optimization.AutoZygote())
    prob = Optimization.OptimizationProblem(optf, p0)
    sol  = Optimization.solve(prob, ConjugateGradient(), maxiters=iterations, callback = callback)

    Zopt, θopt, 𝛃opt, μopt, Λrootopt, wopt, bopt = upk(sol.u, 𝛃)

    VERIFY ? numerically_verify_gplvmplus(X, upk(results.minimizer, 𝛃)..., JITTER, η) : nothing


    #---------------------------------------------------------------------------
    # Retrieve optimised parameters.
    # Calculate optimal prior kernel covariance matrix Kopt and return other
    # optimised variational parameters.
    #---------------------------------------------------------------------------

    Kopt, Σopt = let

        local D² = pairwise(SqEuclidean(), Zopt)

        local K = Symmetric(covariance(D², θopt) + JITTER*I)
    
        local Σ  = aux_invert_K⁻¹_plus_Λ(;K = K, Λroot = Λrootopt) + JITTER*I

        K, Σ
    end

    return (w = wopt, net = net, η = η, Σ = Σopt, Z = Zopt, θ = θopt, 𝛃 = 𝛃opt, μ = μopt, Λroot = Λrootopt, K = Kopt, b = bopt, JITTER = JITTER)

end