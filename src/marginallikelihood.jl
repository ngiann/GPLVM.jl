function marginallikelihood(X, Z, θ, β, μ, Λroot, w, α, b; JITTER = JITTER, η = η)

    local N = size(Λroot, 1); @assert(size(μ, 2) == size(Z, 2) == size(X, 2) == N)

    local D = size(X, 1); @assert(size(μ, 1) == D)

    local D² = pairwise(SqEuclidean(), Z)

    local K  = Symmetric(covariance(D², θ) + JITTER*I)

    local Σ  = woodbury(;K = K, Λ½ = Λroot)+ JITTER*I

    local U = cholesky(K).L
    
    local ℓ = zero(eltype(Z))


    # log-prior contribution - implements equation eq:log_prior_gplvm_pos

    ℓ += - 0.5*sum(abs2.(U\μ'))  + D*(-0.5*N*log(2π)-sum(log.(diag(U)))) - D*0.5*tr(U'\(U\Σ)) # tr(U'\(U\Σ)) equiv to tr(K\Σ)


    # log-likelihood contribution - implements eq:log_likel_gplvm_pos

    local E = exp.(α*μ   .+     α^2*diag(Σ)' / 2 .+  b)

    local B = exp.(2*α*μ .+ (2*α)^2*diag(Σ)' / 2 .+ 2b) 

    local V = B .- E.^2 # this is V[X] = E[X²] - (E[X])² # There  may be a computational gain here

    ℓ += -0.5*D*N*log(2π) + 0.5*D*N*log(β)  -0.5*β*sum(abs2.(myskip.(X .- E))) - β/2 * sum(V)


    # entropy contribution with constants discarded - implements eq:entropy_gplvm_pos

    ℓ += 0.5*D*logdet(Σ) 


    # penalty on latent coordinates - not in latex

    ℓ += - 0.5*η*sum(abs2.(Z))


    # penalty on neural network weights to smoothen optimisation - not in latex

    ℓ += - 0.5*η*sum(abs2.(w))

    return ℓ

end
