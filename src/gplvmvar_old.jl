function gplvmvar(X; iterations = 1, α = 1e-2, seed = 1, Q = 2, JITTER = 1e-6)

    rg = MersenneTwister(seed)

    
    N = size(X, 2)
    D = size(X, 1)

    
    @printf("Running gplvmvar.\n")
    @printf("There are %d number of data items\n", N)
    @printf("There are %d number of dimensions\n", D)
    @printf("Q=%d\n", Q)
    

    #-------------------------------------------
    function unpack(p)
    #-------------------------------------------

        @assert(length(p) == Q*N + 2 + 1 + N*D + N)

        local MARK = 0

        local Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

        local θ = exp.(p[MARK+1:MARK+2]); MARK += 2

        local β = exp(p[MARK+1]); MARK += 1

        local μ  = reshape(p[MARK+1:MARK+D*N], D, N); MARK += D*N

        local Λroot = Diagonal(reshape(p[MARK+1:MARK+N], N)); MARK += N

        @assert(MARK == length(p))

        return Z, θ, β, μ, Λroot

    end


    #-------------------------------------------
    function marginallikelihood(Z, θ, β, μ, Λroot)
    #-------------------------------------------

        local D² = pairwise(SqEuclidean(), Z)

        local K  = Symmetric(covariance(D², θ) + JITTER*I)

        local Σ  = woodbury(;K = K, Λ½ = Λroot)

        local ℓ = zero(eltype(Z))

        local U = cholesky(K).L

        ℓ += - 0.5*sum(abs2.(U\μ'))  + D*(-0.5*N*log(2π)-sum(log.(diag(U))))

        ℓ += - 0.5*β*sum(abs2.(X.-μ)) + D*(0.5*N*log(β) - 0.5*N*log(2π))

        ℓ += D*(0.5*logdet(Σ) - 0.5*tr(U'\(U\Σ)) - 0.5*β*tr(Σ)) # tr(U'\(U\Σ)) is equivalent to tr(K\Σ)


        return ℓ - 0.5*α*sum(abs2.(Z))
    
    end
    

    objective(p) = -marginallikelihood(unpack(p)...)

    p0 = let 
        
        [randn(rg, Q*N)*0.2; randn(rg,3)*1; 0.1*randn(rg, D*N); randn(rg, N)]

    end

    @printf("Optimising %d number of parameters\n",length(p0))

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 2)


    gradhelper!(s, p) = copyto!(s, Zygote.gradient(objective, p)[1])

    results = optimize(objective, gradhelper!, p0, ConjugateGradient(eta=1e-6), opt)

    Zopt, θopt, σ²opt, μopt, Λopt = unpack(results.minimizer)
 

    return Zopt, θopt, σ²opt, μopt, Λopt

end