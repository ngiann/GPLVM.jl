function gplvmvarnet(X, 𝛔 = missing; iterations = 1, η = 1e-2, seed = 1, Q = 2, JITTER = 1e-6,  H1 = 10, H2 = H1, VERIFY = false)
    
    #---------------------------------------------------------------------------
    # Setup variables and free parameters: set random seed, get dimensions, etc
    #---------------------------------------------------------------------------

    # fix random seed for reproducibility

    rg = MersenneTwister(seed)

    # auxiliary type for dispatching to appropriate method

    modeltype = Val(:gplvmvarnet)

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

    net = ThreeLayerNetwork(in = Q, H1 = H1, H2 = H2, out=D)
    
    nwts = ForwardNeuralNetworks.numweights(net)
    
    
    # Initialise free parameters randomly

    p0 = let 
        
        ismissing(𝛃) ? [randn(rg, Q*N)*0.2; randn(rg,3)*1; 0.1*randn(rg, nwts); randn(rg, N)] :
        [randn(rg, Q*N)*0.2; randn(rg,2)*1; 0.1*randn(rg, nwts); randn(rg, N)]
        
    end
    
    # define auxiliary unpack function

    upk(p, 𝛃) = unpack(modeltype, p, 𝛃, D, N, net, Q)


    #---------------------------------------------------------------------------
    # Define optimisation problem
    #---------------------------------------------------------------------------
    
    # setup objective function and gradient

    objective(p) = -marginallikelihood(modeltype, X, upk(p, 𝛃)...; JITTER = JITTER, η = η)
    
    function fg!(F, G, x)
        
        value, ∇f = Zygote.withgradient(objective, x)
        
        isnothing(G) || copyto!(G, ∇f[1])
        
        isnothing(F) || return value
        
        nothing
        
    end

    # set options for optimiser

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    # numerically verify before optimisation

    VERIFY ? numerically_verify(modeltype, X, upk(p0, 𝛃)..., JITTER, η) : nothing

    # run actual optimisation
    
    @printf("Optimising %d number of parameters\n",length(p0))

    results = optimize(Optim.only_fg!(fg!), p0, LBFGS(), opt)

    # numerically verify after optimisation

    VERIFY ? numerically_verify_gplvm_var_net(modeltype, X, results.minimizer, 𝛃, JITTER, η) : nothing


    #---------------------------------------------------------------------------
    # Retrieve optimised parameters.
    # Calculate optimal prior kernel covariance matrix Kopt and return other
    # optimised variational parameters.
    #---------------------------------------------------------------------------

    Zopt, θopt, 𝛃opt, μopt, Λrootopt = upk(results.minimizer, 𝛃)
 
    Kopt = let

        local D² = pairwise(SqEuclidean(), Zopt)

        Symmetric(covariance(D², θopt) + JITTER*I)

    end

    return (Z = Zopt, θ = θopt, 𝛃 = 𝛃opt, μ = μopt, Λroot = Λrootopt, K = Kopt, JITTER = JITTER)

end