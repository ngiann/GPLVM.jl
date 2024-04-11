function marginallikelihood_VERIFY_gplvmplus(X, Z, θ, β, μ, Λroot, w, α, b; JITTER = JITTER, η=η)

    local N = size(Λroot, 1); @assert(size(μ, 2) == size(Z, 2) == size(X, 2) == N)

    local D = size(X, 1); @assert(size(μ, 1) == D)
    
    local D² = pairwise(SqEuclidean(), Z)

    local K  = Symmetric(covariance(D², θ) + JITTER*I)

    local Σ  = aux_invert_K⁻¹_plus_Λ(;K = K, Λroot = Λroot)+ JITTER*I

    local ℓ = zero(eltype(Z))


    # log-prior contribution - implements equation eq:log_prior_gplvm_pos
    let 
        prior = MvNormal(zeros(N), K)
        for d in 1:D
            ℓ += logpdf(prior, μ[d,:]) - 0.5*tr(K\Σ) 
        end
    end


    # log-likelihood contribution - implements eq:log_likel_gplvm_pos
    
    for n in 1:N, d in 1:D
        
        ℓ += logpdf(Normal(E(a = α, μ = μ[d,n], σ = sqrt(Σ[n,n]), b = b), 1/sqrt(β)), X[d,n]) - 
                0.5*β*V(a = α, μ = μ[d,n], σ = sqrt(Σ[n,n]), b = b)
    
    end

    # entropy contribution with constants discarded - implements eq:entropy_gplvm_pos

    ℓ += D * Distributions.entropy(MvNormal(zeros(N), Σ))

    # penalty on latent coordinates - not in latex

    ℓ += - 0.5*η*sum(abs2.(Z))

    # penalty on neural network weights to smoothen optimisation - not in latex

    ℓ += - 0.5*η*sum(abs2.(w))

    return ℓ

end
