function gplvm(X; iterations = 1, α = 1e-2, seed = 1, Q = 2, JITTER = 1e-8)

    # Fix random number generator for reproducibility

    rg = MersenneTwister(seed)

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

    upk(p) = unpack_gplvm(p, D, N, Q)

    objective(p) = -marginallikelihood(upk(p)...)

    
    # Initialise parameters

    paraminit() = [randn(rg, Q*N)*0.1; randn(rg,4)*3]

    # setup optimiser, run optimisation and retrieve optimised parameters

    @printf("Optimising %d number of parameters\n",length(paraminit()))
    optf = Optimization.OptimizationFunction((u,_)->objective(u), Optimization.AutoZygote())
    prob = Optimization.OptimizationProblem(optf, paraminit())
    sol  = Optimization.solve(prob, ConjugateGradient(), maxiters=iterations, callback = callback)

    Zopt, θopt, σ²opt, bopt = upk(sol.u)


    Kopt = let
        
        local D² = pairwise(SqEuclidean(), Zopt)

        Symmetric(covariance(D², θopt) + σ²opt*I + JITTER*I)

    end

    return (Z = Zopt, X = X, θ = θopt, σ² = σ²opt,  b = bopt, JITTER = JITTER, idx = idx, K = Kopt)

end