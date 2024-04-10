function gplvmplus(X, 𝛔 = missing; iterations = 1, H1 = 10, H2 = H1, seed = 1, Q = 2, JITTER = 1e-6, η = 1e-2, VERIFY = false)

    #---------------------------------------------------------------------------
    # Setup variables and free parameters: set random seed, get dimensions, etc
    #---------------------------------------------------------------------------

    # fix random seed for reproducibility

    rg = MersenneTwister(seed)

    # get dimensions of data

    D, N = size(X)
    
    # report to user

    @printf("Running gplvmplus.\n")
    @printf("There are %d number of data items\n", N)
    @printf("There are %d number of dimensions\n", D)
    @printf("Q=%d\n", Q)

    # we work with precisions instead of standard deviations

    𝛃 = inverterrors(𝛔)


    # define neural network that variational parameters and its number of weights

    net = ThreeLayerNetwork(in = Q, H1 = H1, H2 = H2, out=D)
    
    nwts = ForwardNeuralNetworks.numweights(net)
  
    
    # initialise parameters randomly

    p0 = let
        
        ismissing(𝛃) ? [randn(rg, Q*N)*0.2; randn(rg,2)*1; 0.2*randn(rg, nwts); randn(rg, N); randn(rg, 2)] : 
                       [randn(rg, Q*N)*0.2; randn(rg,1)*1; 0.2*randn(rg, nwts); randn(rg, N); randn(rg, 2)]

    end

    # define auxiliary unpack function

    upk(p, 𝛃) = unpack_gplvmplus(p, 𝛃, D, N, net, Q)


    #------------------------------------------------
    # Setup and run optimiser
    #------------------------------------------------

    objective(p) = -marginallikelihood_gplvmplus(X, upk(p, 𝛃)...; JITTER = JITTER, η = η)

    function callback(state,l)
        # callback function to observe training

        @printf("Iter %d, fitness = %4.6f\n", state.iter, l)
        
        return false

    end
    
    VERIFY ? numerically_verify_gplvmplus(X, upk(p0, 𝛃)..., JITTER, η) : nothing
    
    @printf("Optimising %d number of parameters\n",length(p0))
    optf = Optimization.OptimizationFunction((u,_)->objective(u), Optimization.AutoZygote())
    prob = Optimization.OptimizationProblem(optf, p0)
    sol  = Optimization.solve(prob, ConjugateGradient(), maxiters=iterations, callback = callback)

    Zopt, θopt, 𝛃opt, μopt, Λrootopt, wopt, αopt, bopt = upk(sol.u, 𝛃)


    VERIFY ? numerically_verify_gplvmplus(X, upk(results.minimizer, 𝛃)..., JITTER, η) : nothing

    #-----------------------------------------------------------------
    # Carry out actual optimisation and obtain optimised parameters
    #-----------------------------------------------------------------


   
    # results = optimize(Optim.only_fg!(fg!), p0, ConjugateGradient(), opt) # alphaguess = InitialQuadratic(α0=1e-8)

    # Zopt, θopt, 𝛃opt, μopt, Λrootopt, wopt, αopt, bopt = upk(results.minimizer, 𝛃)
   

    #-----------------------------------------------------------------
    # Return optimised latent coordinates and other parameters
    #-----------------------------------------------------------------

    return let 

        local D²    = pairwise(SqEuclidean(), Zopt)

        local Kopt  = Symmetric(covariance(D², θopt) + JITTER*I)

        local Σopt  = aux_invert_K⁻¹_plus_Λ(;K = Kopt, Λroot = Λrootopt) + JITTER*I

        (μ = μopt, Σ = Σopt, K = Kopt, η = η, Λroot = Λrootopt, net = net, w = wopt,
         α = αopt, b = bopt, 𝛃 = 𝛃opt, Z = Zopt, θ = θopt, JITTER = JITTER, rg = rg)
    end

    
    
end