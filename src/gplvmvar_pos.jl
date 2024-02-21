function gplvmvar_pos(X; iterations = 1, H1 = 10, H2 = H1, seed = 1, Q = 2, JITTER = 1e-6, η = 1e-2)

    rg = MersenneTwister(seed)

    D, N = size(X)

    net = ThreeLayerNetwork(in = Q, H1 = H1, H2 = H2, out=D)
    
    nwts = numweights(net)

    @printf("Running gplvmvar_pos.\n")
    @printf("There are %d number of data items\n", N)
    @printf("There are %d number of dimensions\n", D)
    @printf("Q=%d\n", Q)
    
  
    #-------------------------------------------------
    function unpack(p)
    #-------------------------------------------------

        @assert(length(p) == Q*N + 1 + 1 + nwts + N + 2)

        local MARK = 0

        local Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

        local θ = exp(p[MARK+1]); MARK += 1

        local β = exp(p[MARK+1]); MARK += 1

        local w = p[MARK+1:MARK+nwts]; MARK += nwts

        local Λroot = Diagonal(p[MARK+1:MARK+N]); MARK += N

        local α = exp(p[MARK+1]); MARK += 1

        local b = p[MARK+1]; MARK += 1

        @assert(MARK == length(p))

        local μ = net(w, Z)
        
        return Z, [one(eltype(p));θ], β, μ, Λroot, w, α, b

    end


    
    # initialise parameters randomly

    p0 = [randn(rg, Q*N)*0.2; randn(rg,2)*1; 0.2*randn(rg, nwts); randn(rg, N); randn(rg, 2)]


    # let
    #     aaa = marginallikelihood_verify_1(X, unpack(p0)...; JITTER = JITTER, η = η)
    #     bbb =          marginallikelihood(X, unpack(p0)...; JITTER = JITTER, η = η)
    #     @show aaa, bbb, aaa-bbb
    # end


    #-----------------------------------------------------------------
    # define options, loss and gradient to be passed to Optim.optimize
    #-----------------------------------------------------------------

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    objective(p) = -marginallikelihood(X, unpack(p)...; JITTER = JITTER, η = η)

    function fg!(F, G, x)
            
        value, ∇f = Zygote.withgradient(objective,x)

        isnothing(G) || copyto!(G, ∇f[1])

        isnothing(F) || return value

        nothing

    end

    
    #-----------------------------------------------------------------
    # Carry out actual optimisation and obtain optimised parameters
    #-----------------------------------------------------------------

    @printf("Optimising %d number of parameters\n",length(p0))

    results = optimize(Optim.only_fg!(fg!), p0, ConjugateGradient(), opt) # alphaguess = InitialQuadratic(α0=1e-8)

    Zopt, θopt, _βopt, μopt, Λrootopt, _wopt, αopt, bopt = unpack(results.minimizer)
   

    #-----------------------------------------------------------------
    # Return prediction function and optimised latent coordinates
    #-----------------------------------------------------------------

    sampleprediction = let

        local D²    = pairwise(SqEuclidean(), Zopt)

        local Kopt  = Symmetric(covariance(D², θopt) + JITTER*I)

        local Σopt  = woodbury(;K = Kopt, Λ½ = Λrootopt) + JITTER*I

        samplelatent(ztest) = predict(ztest, Zopt, θopt, μopt, Σopt, Kopt; JITTER = JITTER)

        function sample(ztest)

            local aux = samplelatent(ztest)

            () -> exp.(αopt * aux() .+ bopt)

        end

    end

    return Zopt, sampleprediction
    
end