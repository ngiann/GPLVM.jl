function warpedgplvm(X; iterations = 1, α = 1e-2, seed = 1, Q = 2, JITTER = 1e-6, VERIFY = false)

    @assert(all(X .> 0.0))

    rg = MersenneTwister(seed)

    
    N = size(X, 2)
    D = size(X, 1)

    
    @printf("Running warpedgplvm.\n")
    @printf("There are %d number of data items\n", N)
    @printf("There are %d number of dimensions\n", D)
    @printf("Q=%d\n", Q)
    

    #-------------------------------------------
    function unpack(p)
    #-------------------------------------------

        @assert(length(p) == Q*N + 2 + 2)

        local MARK = 0

        local Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

        local θ = exp.(p[MARK+1:MARK+2]); MARK += 2

        local σ² = exp(p[MARK+1]); MARK += 1

        local b = p[MARK+1]; MARK += 1

        @assert(MARK == length(p))

        return Z, θ, σ², b

    end


    #-------------------------------------------
    function noisykernelmatrix_chol(Z, θ, σ²)
    #-------------------------------------------

        local D² = pairwise(SqEuclidean(), Z)

        local K  = Symmetric(covariance(D², θ))

        cholesky(Symmetric(K + (σ²+JITTER)*I)).L

    end
        

    #-------------------------------------------
    function marginallikelihood(Z, θ, σ², b)
    #-------------------------------------------

        # Implements eq. (6) in paper "Warped Gaussian Processes"
        # The function f(⋅) we use here is f(⋅) = log(⋅) + b

        local U = noisykernelmatrix_chol(Z, θ, σ²)

        local ℓ = D * (-0.5*N*log(2π)  - 0.5*2*sum(log.(diag(U)))) -0.5*sum(abs2.((U\(b .+ log.(X')))))

        ℓ += - sum(log.(1.0 ./ X))
            
        return ℓ - 0.5*α*sum(abs2.(Z))

    end


    #-------------------------------------------
    function marginallikelihood_verify(Z, θ, σ²,b)
    #-------------------------------------------
    
        local D² = pairwise(SqEuclidean(), Z)

        local K = Symmetric(covariance(D², θ) + (σ²+JITTER)*I)
    
        local ℓ = 0.0

        for d in 1:D
            
            ℓ += logpdf(MvNormal(zeros(N), K), b.+ log.(X[d,:]))

        end

        ℓ += - sum(log.(1.0 ./ X))
        
        return ℓ - 0.5*α*sum(abs2.(Z))
    
    end


    if VERIFY
        local p0 = [randn(rg, Q*N)*0.1; randn(rg,4)*3]
        @printf("Following two values should be really close to each other:\n")
        local v1 = marginallikelihood_verify(unpack(p0)...)
        local v2 = marginallikelihood(unpack(p0)...)
        @printf("%f\n", v1)
        @printf("%f\n", v2)
        @printf("difference is %f\n", v1-v2)
    end


    objective(p) = -marginallikelihood(unpack(p)...)

    paraminit = let 
        
        local p0 = [randn(rg, Q*N)*0.1; randn(rg,4)*3]

        # local nmopt = Optim.Options(iterations = 1000, show_trace = true, show_every = 10)

        # optimize(objective, p0, NelderMead(), nmopt).minimizer

    end



    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)


    gradhelper!(s, p) = copyto!(s, Zygote.gradient(objective, p)[1])

    results = optimize(objective, gradhelper!, paraminit, LBFGS(), opt)

    Zopt, θopt, σ²opt, bopt = unpack(results.minimizer)

    noisy_K_chol_opt = noisykernelmatrix_chol(Zopt, θopt, σ²opt)
    
    return (Z = Zopt, θ = θopt, β = 1/σ²opt, noisy_K_chol = noisy_K_chol_opt, b = bopt, JITTER = JITTER)

end