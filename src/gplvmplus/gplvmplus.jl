function gplvmplus(X; iterations = 1, H1 = 10, H2 = H1, seed = 1, Q = 2, JITTER = 1e-4, η = 1e-2, VERIFY = false)
    
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

    @printf("Running gplvmplus.\n")
    @printf("There are %d number of data items\n", N)
    @printf("There are %d number of dimensions\n", D)
    @printf("Q=%d\n", Q)


    # define neural network that variational parameters and its number of weights

    net = ThreeLayerNetwork(in = Q, H1 = H1, H2 = H2, out=D)
    
    nwts = ForwardNeuralNetworks.numweights(net)    
  
    
    # initialise parameters randomly

    p0 = [randn(rg, Q*N)*0.1; randn(rg,1)*0.1; log(1.0); 0.2*randn(rg, nwts); 10*ones(N); 0.1*randn(rg, 2); 0.1*randn(rg, N)]

    # define auxiliary unpack function

    upk(p) = unpack_gplvmplus(p, D, N, net, Q)


    #------------------------------------------------
    # Setup and run optimiser
    #------------------------------------------------

    objective(p) = -marginallikelihood_gplvmplus(X, idx, upk(p)...; JITTER = JITTER, η = η)
    
    VERIFY ? numerically_verify_gplvmplus(X, upk(p0)..., JITTER, η) : nothing
    
    @printf("(B) Optimising %d number of parameters\n",length(p0))
    optf = Optimization.OptimizationFunction((u,_)->objective(u), Optimization.AutoZygote())
    prob = Optimization.OptimizationProblem(optf, p0)
    sol  = Optimization.solve(prob, LBFGS(), maxiters=iterations, callback = callback)
    Zopt, θopt,βopt, μopt, Λrootopt, wopt, αopt, bopt, copt = upk(sol.u)
   
    # opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 10)
    # fg! = getfg!(objective)   
    # results = optimize(Optim.only_fg!(fg!), p0, LBFGS(),opt)# BFGS(alphaguess = InitialStatic(scaled=true)), opt)
    # Zopt, θopt,βopt, μopt, Λrootopt, wopt, αopt, bopt = upk(results.minimizer)

    
    # VERIFY ? numerically_verify_gplvmplus(X, upk(results.minimizer)..., JITTER, η) : nothing
    VERIFY ? numerically_verify_gplvmplus(X, upk(sol.u)..., JITTER, η) : nothing


    #-----------------------------------------------------------------
    # Return optimised latent coordinates and other parameters
    #-----------------------------------------------------------------

    return let 

        local D²    = pairwise(SqEuclidean(), Zopt)

        local Kopt  = Symmetric(covariance(D², θopt) + JITTER*I)

        local Σopt  = aux_invert_K⁻¹_plus_Λ(;K = Kopt, Λroot = Λrootopt) + JITTER*I

        (μ = μopt, Σ = Σopt, K = Kopt, η = η, Λroot = Λrootopt, net = net, w = wopt,
         α = αopt, b = bopt, c = copt, β = βopt, Z = Zopt, θ = θopt, JITTER = JITTER, rg = rg)
    end

    
    
end