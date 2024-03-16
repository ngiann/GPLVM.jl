function gplvm(X; iterations = 1, α = 1e-2, seed = 1, Q = 2, JITTER = 1e-6, VERIFY = false)

    # Fix random number generator for reproducibility

    rg = MersenneTwister(seed)

    # auxiliary type for dispatching to appropriate method

    modeltype = Val(:gplvm)

    # Figure our dimensions

    D, N = size(X)

    idx = [findall(x->~isinf(x), X[d,:]) for d in 1:D]
    
    # Report message

    @printf("Running gplvm.\n")
    @printf("There are %d number of data items\n", N)
    @printf("There are %d number of dimensions\n", D)
    @printf("Q=%d\n", Q)


    #-------------------------------------------
    function marginallikelihood(Z, θ, σ², b)
    #-------------------------------------------

        local D² = pairwise(SqEuclidean(), Z)

        local K = Symmetric(covariance(D², θ))

        local ℓ = zero(eltype(Z))

        # marginal likelihood contribution

        for d in 1:D

            Xd  = @views X[d,idx[d]]
            
            Kpartition = @views K[idx[d], idx[d]]

            Kc = cholesky(Kpartition + σ²*I + JITTER*I).L

            ℓ += -0.5*sum(abs2.(Kc\(Xd.-b))) - 0.5*2*sum(log.(diag(Kc))) - 0.5*length(idx[d])*log(2π)
            
            # let
            #     tmp1 = logpdf(MvNormal(zeros(length(idx[d])).+b, Kpartition + σ²*I + JITTER*I), X[d,idx[d]])
            #     tmp2 = -0.5*sum(abs2.(Kc\(Xd.-b))) - 0.5*2*sum(log.(diag(Kc))) - 0.5*length(idx[d])*log(2π)
            #     @show tmp1, tmp2, tmp1-tmp2
            # end
        end
        
        return ℓ - 0.5*α*sum(abs2.(Z)) # penalty on latent coordinates

    end


    # convenience functions

    upk(p) = unpack(modeltype, p, D, N, Q)

    objective(p) = -marginallikelihood(upk(p)...)

    
    # Initialise parameters

    paraminit() = [randn(rg, Q*N)*0.1; randn(rg,4)*3]

    # setup optimiser, run optimisation and retrieve optimised parameters

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    function fg!(F, G, x)
            
        value, ∇f = Zygote.withgradient(objective, x)

        isnothing(G) || copyto!(G, ∇f[1])

        isnothing(F) || return value

        nothing

    end

    
    initialsolutions = [paraminit() for _ in 1:100]

    bestindex = argmin(map(objective, initialsolutions))

    results = optimize(Optim.only_fg!(fg!), initialsolutions[bestindex], LBFGS(), opt)

    Zopt, θopt, σ²opt, bopt = upk(results.minimizer)


    Kopt = let
        
        local D² = pairwise(SqEuclidean(), Zopt)

        Symmetric(covariance(D², θopt) + σ²opt*I + JITTER*I)

    end

    return (Z = Zopt, X = X, θ = θopt, σ² = σ²opt,  b = bopt, JITTER = JITTER, idx = idx, K = Kopt)

end