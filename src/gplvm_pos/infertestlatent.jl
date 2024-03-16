# There are two cases cases:

# 1. At testing no error measurements provided: assume at training GPLVM₊ precision was optimised.

function inferlatent(X₊, R; iterations = 10, repeats=1) 
    
    @assert isa(R[:𝛃], FillArrays.AbstractFillMatrix)

    infertestlatent(X₊, Fill(R[:𝛃][1], size(X₊)); μ = R[:μ], Σ = R[:Σ], K = R[:K], η = R[:η], Λroot = R[:Λroot], net = R[:net], w = R[:w],
    α = R[:α], b = R[:b], Z = R[:Z], θ = R[:θ], JITTER = R[:JITTER], rg = R[:rg], iterations = iterations, repeats = repeats)

end

# 2. At testing error measuments are provided.

inferlatent(X₊, 𝛔, R; iterations = iterations, repeats = repeats) = infertestlatent(X₊, inverterrors(𝛔);  μ = R[:μ], Σ = R[:Σ], K = R[:K], η = R[:η], Λroot = R[:Λroot], net = R[:net], w = R[:w],
α = R[:α], b = R[:b], Z = R[:Z], θ = R[:θ], JITTER = R[:JITTER], rg = R[:rg], iterations = iterations, repeats = repeats)



function infertestlatent(X₊, 𝛃; μ = μ, Σ = Σ, K = K, η = η, Λroot = Λroot, net = net, w = w,
                             α = α, b = b, Z = Z, θ = θ, JITTER = JITTER, rg = rg, iterations = iterations, repeats = repeats)

    # sort out dimensions

    D, N = size(μ); @assert(N == size(Z, 2) == size(Λroot, 1) == size(Λroot, 2))

    Q = size(Z, 1)

    N₊ = size(X₊, 2); @assert(D == size(X₊, 1)); @assert(size(X₊) == size(𝛃))


    # pre-calculate
    
    inv_K_plus_Λ⁻¹ = aux_invert_K_plus_Λ⁻¹(K=K, Λroot=Λroot)

    inv_K_mul_μᵀ = (K\μ')


    #--------------------------------------------------
    function unpack(p)
    #--------------------------------------------------

        local MARK = 0

        local Z₊ = reshape(p[MARK+1:MARK+Q*N₊], Q, N₊); MARK += Q*N₊
        
        local Lroot = Diagonal(p[MARK+1:MARK+N₊]); MARK += N₊
        
        @assert(MARK == length(p)) # all parameters must be used up

        local ν  = net(w, Z₊)

        return Z₊, ν, Lroot

    end


    #--------------------------------------------------
    function objective(Z₊, ν, Lroot)
    #--------------------------------------------------

        # Calculate cross-covariance matrix between test and training inputs
        local K₊ = covariance(pairwise(SqEuclidean(), Z₊, Z), θ); @assert(size(K₊, 1) == N₊)

        # Calculate "self"-covariance matrix between test inputs
        local K₊₊ = Symmetric(covariance(pairwise(SqEuclidean(), Z₊), θ) + JITTER*I); @assert(size(K₊₊, 1) == N₊)
        
        # calculate mean of "prior" of test latent function values
        local m = (K₊*inv_K_mul_μᵀ)'

        # calculate covariance of "prior" of test latent function values
        local C = K₊₊ - K₊*inv_K_plus_Λ⁻¹*K₊'; @assert(size(C, 1) == N₊); @assert(size(C, 2) == N₊);

        local Cᵤ = cholesky(Symmetric(C)).L

        # calculate posterior covariance of test latent function values
        local A = aux_invert_K⁻¹_plus_Λ(K=Symmetric(C+JITTER*I) , Λroot = Lroot)

        
        # log-prior contribution
        local ℓ = -0.5*D*N₊*log(2π) - 0.5*sum(abs2.(Cᵤ\(ν-m)')) - D*0.5*tr(C\A) - D*sum(log.(diag(Cᵤ)))

        # code below implements line above - keep for numerical verification
        # let
        #     ℓ1 = 0
        #     for d in 1:D
        #         ℓ1 += logpdf(MvNormal(ν[d,:], Symmetric(C)), m[d,:]) - 0.5*tr(C\A)
        #     end
        # end

        # log-likelihood contribution
        
        local E, V = expectation_latent_function_values(;α = α, b = b, μ = ν, Σ = A)

        ℓ += -0.5*D*N₊*log(2π) + 0.5*sum(log.(𝛃))  -0.5*sum(𝛃 .* abs2.(myskip.(X₊ .- E))) - 1/2 * sum(𝛃 .* V)


        # entropy contribution with constants discarded
        ℓ += 0.5*D*logabsdet(A)[1] 

        # penalty on latent - not in latex
        ℓ += - 0.5*η*sum(abs2.(Z₊))

    end



    # initialise parameters randomly
    @assert(N₊ == 1)

    function p0()

        local luckyindex = ceil(Int, rand(rg)*(size(Z,2))) 
        
        [Z[:,luckyindex]; randn(rg, N₊)]

    end

    #-----------------------------------------------------------------
    # define options, loss and gradient to be passed to Optim.optimize
    #-----------------------------------------------------------------

    opt = Optim.Options(iterations = iterations, show_trace = false, show_every = 1)

    objective(p) = -objective(unpack(p)...)

    function fg!(F, G, x)
            
        value, ∇f = Zygote.withgradient(objective,x)

        isnothing(G) || copyto!(G, ∇f[1])

        isnothing(F) || return value

        nothing

    end


    #-----------------------------------------------------------------
    # Carry out actual optimisation and obtain optimised parameters
    #-----------------------------------------------------------------

    @printf("Optimising %d number of parameters\n",length(p0()))

    solutions = [optimize(Optim.only_fg!(fg!), p0(), ConjugateGradient(), opt) for _ in 1:repeats] # alphaguess = InitialQuadratic(α0=1e-8)

    bestindex = argmin([s.minimizer for s in solutions])

    Zopt, νopt, Lroot = unpack(solutions[bestindex].minimizer)
   
    return Zopt

end