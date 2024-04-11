function warpedgplvm(X; iterations = 1, α = 1e-2, seed = 1, Q = 2, JITTER = 1e-6, VERIFY = false)

    @assert(all(X .> 0.0))

    rg = MersenneTwister(seed)

    
    N = size(X, 2)
    D = size(X, 1)
    
    idx = [findall(x->~isinf(x), X[d,:]) for d in 1:D]
    
    @printf("Running warpedgplvm.\n")
    @printf("There are %d number of data items\n", N)
    @printf("There are %d number of dimensions\n", D)
    @printf("Q=%d\n", Q)
    

    #-------------------------------------------
    function unpack(p)
    #-------------------------------------------

        @assert(length(p) == Q*N + 4)

        local MARK = 0

        local Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

        local θ = exp.(p[MARK+1:MARK+2]); MARK += 2

        local σ² = exp(p[MARK+1]); MARK += 1

        local b = p[MARK+1]; MARK += 1

        @assert(MARK == length(p))

        return Z, θ, σ², b

    end


    #-------------------------------------------
    function warp(x)
    #-------------------------------------------

        log(x)

    end


    #-------------------------------------------
    function calculateK(Z, θ, σ²)
    #-------------------------------------------

        local D² = pairwise(SqEuclidean(), Z)

        Symmetric(covariance(D², θ) + σ²*I + JITTER*I)

    end

    #-------------------------------------------
    function marginallikelihood(Z, θ, σ², b)
    #-------------------------------------------

        local K = calculateK(Z, θ, σ²)

        local ℓ = zero(eltype(Z))

        for d in 1:D

            Xd = X[d,idx[d]]
            
            Kpartition = K[idx[d], idx[d]]

            Kc = cholesky(Kpartition).L

            local xd = warp.(Xd)
          
            ℓ += -0.5*sum(abs2.(Kc\(xd.-b))) - 0.5*2*sum(log.(diag(Kc))) - 0.5*length(idx[d])*log(2π)
            
            # ℓ += sum(log.(1 ./ Xd)) # this is constant and can be commented out

        end

        return ℓ - 0.5*α*sum(abs2.(Z))

    end


    #-------------------------------------------
    function marginallikelihood_d(d, K, b)
    #-------------------------------------------

        local Xd = X[d,idx[d]]
            
        local Kpartition = K[idx[d], idx[d]]

        local Kc = cholesky(Kpartition).L

        local xd = warp.(Xd)
            
        -0.5*sum(abs2.(Kc\(xd.-b))) - 0.5*2*sum(log.(diag(Kc))) - 0.5*length(idx[d])*log(2π)
      
    end

    #-------------------------------------------
    function parallel_marginallikelihood(Z, θ, σ², b)
    #-------------------------------------------
    
        local K = calculateK(Z, θ, σ²)

        local aux(d) = marginallikelihood_d(d, K, b)

        sum(Transducers.foldxt(+, Map(aux),  1:D))

    end

   

    
    #-------------------------------------------
    function marginallikelihood_verify(Z, θ, σ²,a,b)
    #-------------------------------------------
    
        local D² = pairwise(SqEuclidean(), Z)

        local K = Symmetric(covariance(D², θ) + (σ²+JITTER)*I)
    
        local ℓ = 0.0

        for d in 1:D
            
            ℓ += logpdf(MvNormal(zeros(N), K), b.+ a*log.(X[d,:]))

        end

        ℓ += sum(log.(a ./ X))
        
        return ℓ - 0.5*α*sum(abs2.(Z))
    
    end


    if VERIFY
        local p0 = [randn(rg, Q*N)*0.1; randn(rg,5)*3]
        @printf("Following two values should be really close to each other:\n")
        local v1 = marginallikelihood_verify(unpack(p0)...)
        local v2 = marginallikelihood(unpack(p0)...)
        @printf("%f\n", v1)
        @printf("%f\n", v2)
        @printf("difference is %f\n", v1-v2)
    end


    objective(p) = -parallel_marginallikelihood(unpack(p)...)

    paraminit = [randn(rg, Q*N)*0.1; randn(rg,4)*0.1]
    
  
    @printf("Optimising %d number of parameters\n",length(paraminit))
    optf = Optimization.OptimizationFunction((u,_)->objective(u), Optimization.AutoZygote())
    prob = Optimization.OptimizationProblem(optf, paraminit)
    sol  = Optimization.solve(prob, ConjugateGradient(), maxiters=iterations, callback = callback)

    Zopt, θopt, σ²opt, bopt = unpack(sol.u)

    Kopt = calculateK(Zopt, θopt, σ²opt)

    return (X = X, Z = Zopt, b = bopt, θ = θopt, σ² = σ²opt, K = Kopt, JITTER = JITTER, idx = idx, warp = warp)

end