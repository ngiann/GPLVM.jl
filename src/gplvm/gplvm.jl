function gplvm(X, 𝛔=zeros(size(X)); iterations = 1, α = 1e-2, seed = 1, Q = 2, JITTER = 1e-6, VERIFY = false)

    # Fix random number generator for reproducibility

    rg = MersenneTwister(seed)

    # auxiliary type for dispatching to appropriate method

    modeltype = Val(:gplvm)

    # Figure our dimensions

    D, N = size(X)

    𝛔² = 𝛔.^2
    
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

            Xd  = @view X[d,:]
            𝛔²d = @view 𝛔²[d,:]

            Kc = cholesky(K + Diagonal(𝛔²d) + σ²*I + JITTER*I).L

            ℓ += -0.5*sum(abs2.(Kc\(Xd.-b))) - 0.5*2*sum(log.(diag(Kc))) - 0.5*N*log(2π)
            
            # let
            #     tmp1 = logpdf(MvNormal(zeros(N).+b, K + Diagonal(𝛔²[d,:]) + JITTER*I), X[d,:])
            #     tmp2 = -0.5*sum(abs2.(Kc\(Xd.-b))) - 0.5*2*sum(log.(diag(Kc))) - 0.5*N*log(2π)
            #     @show tmp1, tmp2, tmp1-tmp2
            # end
        end
        
        return ℓ - 0.5*α*sum(abs2.(Z)) # penalty on latent coordinates

    end


    # convenience functions

    upk(p) = unpack(modeltype, p, D, N, Q)

    objective(p) = -marginallikelihood(upk(p)...)

    
    # Initialise parameters

    paraminit = let 
        
        local p0 = [randn(rg, Q*N)*0.1; randn(rg,4)*3]
 
        local nmopt = Optim.Options(iterations = 500, show_trace = true, show_every = 100)

        optimize(objective, p0, NelderMead(), nmopt).minimizer

    end


    # setup optimiser, run optimisation and retrieve optimised parameters

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    function fg!(F, G, x)
            
        value, ∇f = Zygote.withgradient(objective, x)

        isnothing(G) || copyto!(G, ∇f[1])

        isnothing(F) || return value

        nothing

    end

    results = optimize(Optim.only_fg!(fg!), paraminit, ConjugateGradient(), opt)

    Zopt, θopt, σ²opt, bopt = upk(results.minimizer)


    # return (Z = Zopt, θ = θopt, 𝛔² = 𝛔²opt, noisy_K_chol = noisy_K_chol_opt, b = bopt, JITTER = JITTER)

end