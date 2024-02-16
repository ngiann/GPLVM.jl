function gplvmvar_pos(X; iterations = 1, H = 10, α = 1e-2, seed = 1, Q = 2, JITTER = 1e-6, η = 1e-2)

    rg = MersenneTwister(seed)

    N = size(X, 2)
    D = size(X, 1)

    net = ThreeLayerNetwork(in = Q, H1=H, H2=H, out=D)

    nwts = numweights(net)

    @printf("Running gplvmvar_pos.\n")
    @printf("There are %d number of data items\n", N)
    @printf("There are %d number of dimensions\n", D)
    @printf("Q=%d\n", Q)
    
  
    #-------------------------------------------------
    function unpack(p)
    #-------------------------------------------------

        @assert(length(p) == Q*N + 2 + 1 + nwts + N + 2)

        local MARK = 0

        local Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

        local θ = exp.(p[MARK+1:MARK+2]); MARK += 2

        local β = exp(p[MARK+1]); MARK += 1

        local w = p[MARK+1:MARK+nwts]; MARK += nwts

        local Λroot = Diagonal(p[MARK+1:MARK+N]); MARK += N

        local α = exp(p[MARK+1]); MARK += 1

        local b = p[MARK+1]; MARK += 1

        @assert(MARK == length(p))

        local μ = net(w, Z)
        
        return Z, θ, β, μ, Λroot, w, α, b

    end


    #-------------------------------------------------
    function marginallikelihood(Z, θ, β, μ, Λroot, w, α, b)
    #-------------------------------------------------

        local D² = pairwise(SqEuclidean(), Z)

        local K  = Symmetric(covariance(D², θ) + JITTER*I)

        local Σ  = woodbury(;K = K, Λ½ = Λroot)

        local U = cholesky(K).L
        
        local ℓ = zero(eltype(Z))

        # log-prior contribution

        ℓ += - 0.5*sum(abs2.(U\μ'))  + D*(-0.5*N*log(2π)-sum(log.(diag(U)))) - D*0.5*tr(U'\(U\Σ)) # tr(U'\(U\Σ)) equiv to tr(K\Σ)

        # log-likelihood contribution
        
        for n in 1:N
            for d in 1:D
            
             @inbounds ℓ += Elognormal(y=X[d,n], a = α, μ = μ[d,n], σ = sqrt(Σ[n,n]), β = β, b = b)
            
            end
        end
        

        # ℓ += - 0.5*β*sum(abs2.(X.-μ)) + D*(0.5*N*log(β) - 0.5*N*log(2π)) - D*0.5*β*tr(Σ)

        #---------- VERIFICATION ----------
        # let 
        #    V1 = - 0.5*β*sum(abs2.(X.-μ)) + D*(0.5*N*log(β) - 0.5*N*log(2π)) - D*0.5*β*tr(Σ)
        #    V2 = 0.0
        #    for d in 1:D
        #         V2 += logpdf(MvNormal(μ[d,:],sqrt(1/β)), X[d,:]) - 0.5*β*tr(Σ)
        #    end
        #    @show V1, V2, abs(V1-V2)
        #    @assert(1==2)
        # end
        #-----------------------------------
        
        
        # entropy contribution

        ℓ += D*0.5*logdet(Σ) 

        # penalty on latent coordinates

        ℓ += - 0.5*α*sum(abs2.(Z))

        # penalty on neural network weights

        ℓ += - 0.5*η*sum(abs2.(w))

        return ℓ

    end


    objective(p) = -marginallikelihood(unpack(p)...)


    p0 = [randn(rg, Q*N)*0.2; randn(rg,3)*1; 0.2*randn(rg, nwts); randn(rg, N); randn(rg, 2)]

    @printf("Optimising %d number of parameters\n",length(p0))

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 2)

    function fg!(F, G, x)
            
        value, ∇f = Zygote.withgradient(objective,x)

        isnothing(G) || copyto!(G, ∇f[1])

        isnothing(F) || return value

        nothing

    end

    results = optimize(Optim.only_fg!(fg!), p0, LBFGS(), opt)

    Zopt, θopt, σ²opt, μopt, Λopt = unpack(results.minimizer)
   
    return Zopt, θopt, σ²opt, μopt, Λopt

end