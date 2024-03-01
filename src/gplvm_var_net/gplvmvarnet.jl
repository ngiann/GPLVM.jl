function gplvmvarnet(X, 𝛔 = missing; iterations = 1, η = 1e-2, seed = 1, Q = 2, JITTER = 1e-6,  H1 = 10, H2 = H1, VERIFY = false)

    rg = MersenneTwister(seed)

    𝛃 = inverterrors(𝛔)

    D, N = size(X)
    
    net = ThreeLayerNetwork(in = Q, H1 = H1, H2 = H2, out=D)
    
    nwts = ForwardNeuralNetworks.numweights(net)
    
    @printf("Running gplvmvarnet.\n")
    @printf("There are %d number of data items\n", N)
    @printf("There are %d number of dimensions\n", D)
    @printf("Q=%d\n", Q)
    

    #-------------------------------------------
    function unpack(p, ::Missing)
    #-------------------------------------------

        @assert(length(p) == Q*N + 2 + 1 + nwts + N)

        local MARK = 0

        local Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

        local θ = exp.(p[MARK+1:MARK+2]); MARK += 2

        local β = exp(p[MARK+1]); MARK += 1

        local w = p[MARK+1:MARK+nwts]; MARK += nwts

        local Λroot = Diagonal(reshape(p[MARK+1:MARK+N], N)); MARK += N

        @assert(MARK == length(p))

        local μ = net(w, Z)

        return Z, θ, Fill(β, D, N), μ, Λroot

    end


    #-------------------------------------------
    function unpack(p, 𝛃)
    #-------------------------------------------
    
        @assert(length(p) == Q*N + 2 + nwts + N)

        local MARK = 0

        local Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

        local θ = exp.(p[MARK+1:MARK+2]); MARK += 2

        local w = p[MARK+1:MARK+nwts]; MARK += nwts

        local Λroot = Diagonal(reshape(p[MARK+1:MARK+N], N)); MARK += N

        @assert(MARK == length(p))

        local μ = net(w, Z)

        return Z, θ, 𝛃, μ, Λroot

    end


    
    # Initialise parameters randomly

    p0 = let 
        
        ismissing(𝛃) ? [randn(rg, Q*N)*0.2; randn(rg,3)*1; 0.1*randn(rg, nwts); randn(rg, N)] :
        [randn(rg, Q*N)*0.2; randn(rg,2)*1; 0.1*randn(rg, nwts); randn(rg, N)]
        
    end
    
    @printf("Optimising %d number of parameters\n",length(p0))
    
    objective(p) = -marginallikelihood_gplvm_var_net(X, unpack(p, 𝛃)...; JITTER = JITTER, η = η)

    # numerically verify before optimisation

    if VERIFY
        tmp1 = marginallikelihood_gplvm_var_net(X, unpack(p0, 𝛃)...; JITTER = JITTER, η = η)
        tmp2 = marginallikelihood_gplvm_var_net_VERIFY(X, unpack(p0, 𝛃)...; JITTER = JITTER, η = η)
        @printf("Verifiying calculations\n")
        @printf("First implementation delivers  %f\n", tmp1)
        @printf("Second implementation delivers %f\n", tmp2)
        @printf("difference is %f\n", tmp1-tmp2)
    end

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    function fg!(F, G, x)
            
        value, ∇f = Zygote.withgradient(objective, x)

        isnothing(G) || copyto!(G, ∇f[1])

        isnothing(F) || return value

        nothing

    end

    results = optimize(Optim.only_fg!(fg!), p0, LBFGS(), opt)

    # numerically verify after optimisation

    if VERIFY
        tmp1 =        marginallikelihood_gplvm_var_net(X, unpack(p0, 𝛃)...; JITTER = JITTER, η = η)
        tmp2 = marginallikelihood_gplvm_var_net_VERIFY(X, unpack(p0, 𝛃)...; JITTER = JITTER, η = η)
        @printf("Verifiying calculations\n")
        @printf("First implementation delivers  %f\n", tmp1)
        @printf("Second implementation delivers %f\n", tmp2)
        @printf("difference is %f\n", tmp1-tmp2)
    end

    Zopt, θopt, 𝛃opt, μopt, Λrootopt = unpack(results.minimizer, 𝛃)
 
    Kopt = let

        local D² = pairwise(SqEuclidean(), Zopt)

        Symmetric(covariance(D², θopt) + JITTER*I)

    end

    return (Z = Zopt, θ = θopt, 𝛃 = 𝛃opt, μ = μopt, Λroot = Λrootopt, K = Kopt, JITTER = JITTER)

end