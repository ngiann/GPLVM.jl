function marginallikelihood(::Val{:gplvmvarnet}, X, Z, θ, 𝛃, μ, Λroot; JITTER = JITTER, η = η)

    # sort out dimensions

    local N = size(Z, 2)

    local D = size(μ, 1); @assert(size(μ, 2) == N)

    # calculate prior and posterior covariance matrices K and Σ

    local D² = pairwise(SqEuclidean(), Z)

    local K  = Symmetric(covariance(D², θ) + JITTER*I)

    local Σ  = aux_invert_K⁻¹_plus_Λ(;K = K, Λroot = Λroot) + JITTER*I


    # contribution of prior to marginal log likelihood

    local ℓ = expectation_of_sum_D_log_prior_zero_mean(K=K; μ = μ, Σ = Σ)

    # contribution of likelihood
    
    countObs = count(x -> ~ismissing(x), X)

    ℓ += - 0.5*sum(myskip.(𝛃) .* abs2.(myskip.((X.-μ)))) + 0.5*sum(myskip.(log.(𝛃))) - 0.5*countObs*log(2π) - 0.5*sum(𝛃*diag(Σ))

    # contribution of entropy 

    ℓ += D*entropy(Σ) # note multiplication with D
    

    return ℓ - 0.5*η*sum(abs2.(Z)) # penalty on latent coordinates - not in latex document

end
