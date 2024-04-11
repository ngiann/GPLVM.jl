function marginallikelihood_gplvmplus(X, Z, θ, β, μ, Λroot, w, α, b; JITTER = JITTER, η = η)

    # sort out and verify dimensions

    N = size(Λroot, 1); @assert(size(μ, 2) == size(Z, 2) == size(X, 2) == N)

    D = size(X, 1); @assert(size(μ, 1) == D)

    
    # calculate prior and posterior covariance matrices K and Σ

    D² = pairwise(SqEuclidean(), Z)

    K  = Symmetric(covariance(D², θ) + JITTER*I)

    Σ  = aux_invert_K⁻¹_plus_Λ(;K = K, Λroot = Λroot) + JITTER*I


    # log-prior contribution - implements equation eq:log_prior_gplvm_pos

    ℓ = expectation_of_sum_D_log_prior_zero_mean(;K = K, μ = μ, Σ = Σ) # ❗check what happens with missing observations.


    # log-likelihood contribution - implements eq:log_likel_gplvm_pos - ignores/skips missing data values

    E, V = expectation_latent_function_values(;α = α, b = b, μ = μ, Σ = Σ)

    idx      = findall(x -> ~isinf(x), X)
    countObs = length(idx)

    ℓ += - 0.5*countObs*log(2π) + 0.5*countObs*log(β) - 0.5*β*sum(abs2.((X[idx] - E[idx]))) - 1/2 * β*sum(V[idx])


    # entropy contribution - implements eq:entropy_gplvm_pos

    ℓ += D*entropy(Σ) # note multiplication with D


    # penalty on latent coordinates - not in latex

    ℓ += - 0.5*η*sum(abs2.(Z))


    # penalty on neural network weights to smoothen optimisation - not in latex

    ℓ += - 0.5*η*sum(abs2.(w))

    return ℓ

end
