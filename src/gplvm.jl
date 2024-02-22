function gplvm(X; iterations = 1, α = 1e-2, seed = 1, Q = 2, JITTER = 1e-6)

    rg = MersenneTwister(seed)

    
    N = size(X, 2)
    D = size(X, 1)

    
    @printf("Running gplvm.\n")
    @printf("There are %d number of data items\n", N)
    @printf("There are %d number of dimensions\n", D)
    @printf("Q=%d\n", Q)
    

    #-------------------------------------------
    function unpack(p)
    #-------------------------------------------

        @assert(length(p) == Q*N + 2 + 1)

        local MARK = 0

        local Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

        local θ = exp.(p[MARK+1:MARK+2]); MARK += 2

        local σ² = exp(p[MARK+1]); MARK += 1

        @assert(MARK == length(p))

        return Z, θ, σ²

    end


    #-------------------------------------------
    function marginallikelihood(Z, θ, σ²)
    #-------------------------------------------

        local D² = pairwise(SqEuclidean(), Z)

        local K  = Symmetric(covariance(D², θ))

        local U  = cholesky(Symmetric(K + (σ²+JITTER)*I)).L
        
        # local ℓ = zero(eltype(Z))

        # for d in 1:D

        #     Xd = @views X[d,:]
            
        #     ℓ += -0.5*N*log(2π)  -0.5*sum(abs2.(U\Xd)) - 0.5*2*sum(log.(diag(U)))

        #     # let
        #     #     aaa=logpdf(MvNormal(zeros(N), K), X[d,:])
        #     #     bbb=-0.5*N*log(2π) -0.5*sum(abs2.(U\Xd)) - 0.5*2*sum(log.(diag(U)))
        #     #     @show aaa bbb (aaa-bbb)
        #     #     @assert(5==555)
        #     # end
        # end


        local ℓ = D * (-0.5*N*log(2π)  - 0.5*2*sum(log.(diag(U)))) -0.5*sum(abs2.(U\X'))

        return ℓ - 0.5*α*sum(abs2.(Z))

    end


    objective(p) = -marginallikelihood(unpack(p)...)

    paraminit = let 
        
        local p0 = [randn(rg, Q*N)*0.1; randn(rg,3)*3]

        # local nmopt = Optim.Options(iterations = 1000, show_trace = true, show_every = 10)

        # optimize(objective, p0, NelderMead(), nmopt).minimizer

    end



    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)


    gradhelper!(s, p) = copyto!(s, Zygote.gradient(objective, p)[1])

    results = optimize(objective, gradhelper!, paraminit, ConjugateGradient(eta=0.01), opt)

    Zopt, θopt, σ²opt = unpack(results.minimizer)
    
    return Zopt, θopt, σ²opt

end