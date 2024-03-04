function marginallikelihood(::Val{:gplvmvarnet_pos}, X, Z, θ, 𝛃, μ, Λroot, w, α, b; JITTER = JITTER, η = η)

    # sort out and verify dimensions

    N = size(Λroot, 1); @assert(size(μ, 2) == size(Z, 2) == size(X, 2) == N)

    D = size(X, 1); @assert(size(μ, 1) == D)

    
    # calculate prior and posterior covariance matrices K and Σ

    D² = pairwise(SqEuclidean(), Z)

    K  = Symmetric(covariance(D², θ) + JITTER*I)

    Σ  = aux_invert_K⁻¹_plus_Λ(;K = K, Λroot = Λroot) + JITTER*I


    # log-prior contribution - implements equation eq:log_prior_gplvm_pos

    ℓ = expectation_of_sum_D_log_prior_zero_mean(;K = K, μ = μ, Σ = Σ) # ❗check what happens with missing observations.


    # log-likelihood contribution - implements eq:log_likel_gplvm_pos

    E, V = expectation_latent_function_values(;α = α, b = b, μ = μ, Σ = Σ)

    countObs = count(x -> ~ismissing(x), X)

    ℓ += - 0.5*countObs*log(2π) + 0.5*sum(myskip.(log.(𝛃))) - 0.5*sum(𝛃 .* abs2.(myskip.(X .- E))) - 1/2 * sum(myskip.(𝛃 .* V))


    # entropy contribution with constants discarded - implements eq:entropy_gplvm_pos

    ℓ += D*entropy(Σ) # note multiplication with D


    # penalty on latent coordinates - not in latex

    ℓ += - 0.5*η*sum(abs2.(Z))


    # penalty on neural network weights to smoothen optimisation - not in latex

    ℓ += - 0.5*η*sum(abs2.(w))

    return ℓ

end
