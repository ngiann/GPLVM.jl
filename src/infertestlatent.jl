inferlatent(X₊, R) = infertestlatent(X₊; R...)

function infertestlatent(X₊; μ = μ, Σ = Σ, K = K, η = η, Λroot = Λroot, net = net, w = w,
                             α = α, b = b, β = β, Z = Z, θ = θ, JITTER = JITTER, rg = rg)

    # work out and verify dimensions
    D, N = size(μ); @assert(N == size(Z, 2) == size(Λroot, 1) == size(Λroot, 2))

    Q = size(Z, 1)

    N₊ = size(X₊, 2); @assert(D == size(X₊, 1))


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
        local m = (K₊*(K\μ'))'; #****************************************
       
        @assert(size(m, 1) == D); @assert(size(m, 2) == N₊)

        # calculate covariance of "prior" of test latent function values
        local C = K₊₊ - K₊*aux_invert_K_plus_Λ⁻¹(K=K, Λroot=Λroot)*K₊'; @assert(size(C, 1) == N₊); @assert(size(C, 2) == N₊);


        # calculate posterior covariance of test latent function values
        local A = aux_invert_K⁻¹_plus_Λ(K=Symmetric(C+JITTER*I) , Λroot = Lroot)

        # log-prior contribution
        local ℓ = zero(eltype(Z₊))

        local Cᵤ = cholesky(Symmetric(C)).L
        
        ℓ += -0.5*D*N₊*log(2π) - 0.5*sum(abs2.(Cᵤ\(ν-m)')) - D*0.5*tr(C\A) - D*sum(log.(diag(Cᵤ)))

        # code below implements line above - keep for numerical verification
        # let
        #     ℓ1 = 0
        #     for d in 1:D
        #         ℓ1 += logpdf(MvNormal(ν[d,:], Symmetric(C)), m[d,:]) - 0.5*tr(C\A)
        #     end
        # end

        # log-likelihood contribution

        local E = exp.(α*ν   .+     α^2*diag(A)' / 2 .+  b)

        local B = exp.(2*α*ν .+ (2*α)^2*diag(A)' / 2 .+ 2b) 

        local V = B .- E.^2 # this is V[X] = E[X²] - (E[X])² # There may be a computational gain here

        ℓ += -0.5*D*N₊*log(2π) + 0.5*D*N₊*log(β)  -0.5*β*sum(abs2.(myskip.(X₊ .- E))) - β/2 * sum(V)


        # entropy contribution with constants discarded

        ℓ += 0.5*D*logabsdet(A)[1] 

        # penalty on latent - not in latex

        ℓ += - 0.5*η*sum(abs2.(Z₊))

    end



    # initialise parameters randomly

    p0 = [randn(rg, Q*N₊)*0.2; randn(rg, N₊)]

    #-----------------------------------------------------------------
    # define options, loss and gradient to be passed to Optim.optimize
    #-----------------------------------------------------------------

    opt = Optim.Options(iterations = 1000, show_trace = true, show_every = 1)

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

    @printf("Optimising %d number of parameters\n",length(p0))

    results = optimize(Optim.only_fg!(fg!), p0, ConjugateGradient(), opt) # alphaguess = InitialQuadratic(α0=1e-8)

    Zopt, νopt, Lroot = unpack(results.minimizer)
   
    return Zopt
end