function gplvmvar(X; iterations = 1, η = 1e-2, ξ = 0.1, seed = 1, Q = 2, JITTER = 1e-6,  H1 = 10, H2 = H1, VERIFY = false)
    
    notinf(x) = ~isinf(x)

    idx = findall(notinf.(X))

    @show length(idx)/length(X)

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

    # 𝛃 = inverterrors(𝛔)


    # define neural network that modeltypes variational parameters and its number of weights

    net = ThreeLayerNetwork(in = Q, H1 = H1, H2 = H2, out = D)
    
    nwts = ForwardNeuralNetworks.numweights(net)
    
    
    # Initialise free parameters randomly

    p0 = [randn(rg, Q*N)*0.2; randn(rg,3)*1; 0.1*randn(rg, nwts); randn(rg, N); randn(rg)]
    
    # define auxiliary unpack function

    upk(p) = unpack_gplvmvar(p, D, N, net, Q)


    #---------------------------------------------------------------------------
    # Define optimisation problem
    #---------------------------------------------------------------------------
    
    # setup objective function and gradient

    objective(p) = -marginallikelihood_gplvmvar(X, idx, upk(p)...; JITTER = JITTER, η = η, ξ = ξ)
    
    # VERIFY ? numerically_verify_gplvmplus(X, upk(p0)..., JITTER, η) : nothing
    
    @printf("Optimising %d number of parameters\n",length(p0))
    optf = Optimization.OptimizationFunction((u,_)->objective(u), Optimization.AutoZygote())
    prob = Optimization.OptimizationProblem(optf, p0)
    sol  = Optimization.solve(prob, LBFGS(), maxiters=iterations, callback = callback)

    Zopt, θopt, βopt, μopt, Λrootopt, wopt, bopt = upk(sol.u)

    # VERIFY ? numerically_verify_gplvmplus(X, upk(results.minimizer)..., JITTER, η) : nothing


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

    return (w = wopt, net = net, η = η, Σ = Σopt, Z = Zopt, θ = θopt, β = βopt, μ = μopt, Λroot = Λrootopt, K = Kopt, b = bopt, JITTER = JITTER)

end