function expectation_of_sum_D_log_prior_zero_mean(;K = K, μ = μ, Σ = Σ)

    N = size(Σ, 1); @assert(size(Σ, 2) == N) # make sure it is square
                    @assert(size(Σ) == size(K))
    D = size(μ, 1); @assert(size(μ, 2) == N)

    U = cholesky(K).L

    # tr(U'\(U\Σ)) is equivalent to tr(K\Σ)
    # - sum(log.(diag(U))) is equivalent to -0.5*logdet(K)
    # U\μ' is equivalent to ...

    - 0.5*sum(abs2.(U\μ')) - 0.5*D*N*log(2π) - D*sum(log.(diag(U))) - 0.5*D*tr(U'\(U\Σ))

end