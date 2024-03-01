function marginallikelihood_gplvm_var_net_VERIFY(X, Z, θ, 𝛃, μ, Λroot; JITTER = JITTER, η = η)
    
    # sort out dimensions

    local N = size(Z, 2)

    local D = size(μ, 1); @assert(size(μ, 2) == N)


    # calculate prior and posterior covariance matrices K and Σ

    local D² = pairwise(SqEuclidean(), Z)

    local K  = Symmetric(covariance(D², θ) + JITTER*I)

    local Σ  = aux_invert_K⁻¹_plus_Λ(;K = K, Λroot = Λroot) + JITTER*I


    # accummulate here marginal log likelihood

    local ℓ = zero(eltype(Z))


    # contribution of prior

    for d in 1:D
        
        ℓ += logpdf(MvNormal(zeros(N), K), μ[d,:]) - 0.5*tr(K\Σ)

    end
    

    # contribution of likelihood

    for d in 1:D
        
        for n in 1:N

            ℓ += logpdf(Normal(μ[d,n], 1/sqrt(𝛃[d,n])), X[d,n])

        end

        ℓ += - 0.5*tr(Diagonal(𝛃[d,:])*Σ)

    end

  
    # entropy contribution

    ℓ += 0.5*D*logdet(Σ)
        
    
    return ℓ - 0.5*η*sum(abs2.(Z)) # penalty on latent coordinates
    
end
    