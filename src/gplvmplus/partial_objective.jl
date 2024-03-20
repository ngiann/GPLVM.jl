function partial_objective(Z₊, ν, Lroot; Z = Z, θ = θ, JITTER = JITTER, η = η, D = D, inv_K_plus_Λ⁻¹ = inv_K_plus_Λ⁻¹, inv_K_mul_μ = inv_K_mul_μ)

    N₊ = size(Z₊, 2)

    # Calculate cross-covariance matrix between test and training inputs
    K₊ = covariance(pairwise(SqEuclidean(), Z, Z₊), θ); @assert(size(K₊, 2) == N₊)

    # Calculate "self"-covariance matrix between test inputs
    K₊₊ = Symmetric(covariance(pairwise(SqEuclidean(), Z₊), θ) + JITTER*I); @assert(size(K₊₊, 1) == N₊)
    
    # calculate mean of "prior" of test latent function values
    m = (K₊'*inv_K_mul_μ)'

    # calculate covariance of "prior" of test latent function values
    C = K₊₊ - K₊'*inv_K_plus_Λ⁻¹*K₊; @assert(size(C, 1) == N₊); @assert(size(C, 2) == N₊);
    
    # calculate posterior covariance of test latent function values
    A = aux_invert_K⁻¹_plus_Λ(K=Symmetric(C+JITTER*I) , Λroot = Lroot)
    
    # log-prior contribution
    ℓ = expectation_of_sum_D_log_prior_zero_mean(;K = C, μ = (ν-m), Σ = A)
        
    # # code below implements line above - keep for numerical verification
    # let
    #     ℓ1 = 0
    #     for d in 1:D
    #         ℓ1 += logpdf(MvNormal(ν[d,:], Symmetric(C)), m[d,:]) - 0.5*tr(C\A)
    #     end
    #     @show ℓ1
    # end

    # entropy contribution, note multiplication with D
    ℓ += D*entropy(A)
    
    # penalty on latent - not in latex
    ℓ += - 0.5*η*sum(abs2.(Z₊))

    return ℓ, A

end