function marginallikelihood_gplvmvar(X, idx, Z, θ, β, μ, Λroot, w, b; JITTER = JITTER, η = η, ξ = ξ)

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
    
    # @views ℓ += - 0.5*sum(𝛃[idx] .* abs2.(((X[idx].-μ[idx].-b))))- 0.5*sum(𝛃*diag(Σ)) # + 0.5*sum((log.(𝛃))) 
    @views ℓ += - 0.5*sum(β .* abs2.(((X[idx].-μ[idx].-b))))- 0.5*sum(β*diag(Σ))  + 0.5*length(idx)*log(β)

    # contribution of entropy 

    ℓ += D*entropy(Σ) # note multiplication with D
    
    ℓ += - 0.5*η*sum(abs2.(Z)) # penalty on latent coordinates - not in latex document

    ℓ += - 0.5*ξ*sum(abs2.(w)) # penalty on network weights - not in latex document

    return ℓ 
end
