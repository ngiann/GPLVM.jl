function marginallikelihood_verify_1(X, Z, θ, β, μ, Λroot, w, α, b; JITTER = JITTER, η=η)

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
    
    for n in 1:N, d in 1:D
        # ℓ += Elognormal(y=X[d,n], a = α, μ = μ[d,n], σ = sqrt(Σ[n,n]), β = β, b = b)
        ℓ += Elognormal_barrier(X[d,n]; a = α, μ = μ[d,n], σ = sqrt(Σ[n,n]), β = β, b = b)
    end

    # entropy contribution with constants discarded - implements eq:entropy_gplvm_pos

    ℓ += 0.5*D*logdet(Σ) 

    # penalty on latent coordinates - not in latex

    ℓ += - 0.5*η*sum(abs2.(Z))

    # penalty on neural network weights to smoothen optimisation - not in latex

    ℓ += - 0.5*η*sum(abs2.(w))

    return ℓ

end
