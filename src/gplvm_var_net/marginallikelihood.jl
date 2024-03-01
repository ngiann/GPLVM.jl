function marginallikelihood(::Val{:gplvmvarnet}, X, Z, θ, 𝛃, μ, Λroot; JITTER = JITTER, η = η)

    # sort out dimensions

    local N = size(Z, 2)

    local D = size(μ, 1); @assert(size(μ, 2) == N)

    # calculate prior and posterior covariance matrices K and Σ

    local D² = pairwise(SqEuclidean(), Z)

    local K  = Symmetric(covariance(D², θ) + JITTER*I)

    local Σ  = aux_invert_K⁻¹_plus_Λ(;K = K, Λroot = Λroot) + JITTER*I

    local U = cholesky(K).L
    
    # accummulate here marginal log likelihood

    local ℓ = zero(eltype(Z))

    # contribution of prior - tr(U'\(U\Σ)) is equivalent to tr(K\Σ)

    ℓ += - 0.5*sum(abs2.(U\μ'))  + D*(-0.5*N*log(2π)-sum(log.(diag(U))) - 0.5*tr(U'\(U\Σ)))

    # contribution of likelihood
    
    ℓ += - 0.5*sum(𝛃 .* abs2.(myskip.((X.-μ)))) + 0.5*sum(log.(𝛃)) - 0.5*D*N*log(2π) - 0.5*sum(𝛃*diag(Σ))

    # contribution of entropy 

    ℓ += 0.5*D*logabsdet(Σ)[1]
    

    return ℓ - 0.5*η*sum(abs2.(Z)) # penalty on latent coordinates - not in latex document

end
