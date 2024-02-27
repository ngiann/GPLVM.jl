function gplvmvarnet(X; iterations = 1, α = 1e-2, seed = 1, Q = 2, JITTER = 1e-6,  H1 = 10, H2 = H1)

    rg = MersenneTwister(seed)

    
    N = size(X, 2)
    D = size(X, 1)

    net = ThreeLayerNetwork(in = Q, H1 = H1, H2 = H2, out=D)
    
    nwts = ForwardNeuralNetworks.numweights(net)
    
    @printf("Running gplvmvarnet.\n")
    @printf("There are %d number of data items\n", N)
    @printf("There are %d number of dimensions\n", D)
    @printf("Q=%d\n", Q)
    

    #-------------------------------------------
    function unpack(p)
    #-------------------------------------------

        @assert(length(p) == Q*N + 2 + 1 + nwts + N)

        local MARK = 0

        local Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

        local θ = exp.(p[MARK+1:MARK+2]); MARK += 2

        local β = exp(p[MARK+1]); MARK += 1

        local w = p[MARK+1:MARK+nwts]; MARK += nwts

        local Λroot = Diagonal(reshape(p[MARK+1:MARK+N], N)); MARK += N

        @assert(MARK == length(p))

        local μ = net(w, Z)

        return Z, θ, β, μ, Λroot

    end


    #-------------------------------------------
    function marginallikelihood(Z, θ, β, μ, Λroot)
    #-------------------------------------------

        local D² = pairwise(SqEuclidean(), Z)

        local K  = Symmetric(covariance(D², θ) + JITTER*I)

        local Σ  = aux_invert_K⁻¹_plus_Λ(;K = K, Λroot = Λroot) + JITTER*I

        local ℓ = zero(eltype(Z))

        local U = cholesky(K).L

        # contribution of prior - tr(U'\(U\Σ)) is equivalent to tr(K\Σ)

        ℓ += - 0.5*sum(abs2.(U\μ'))  + D*(-0.5*N*log(2π)-sum(log.(diag(U))) - 0.5*tr(U'\(U\Σ)))

        # contribution of likelihood

        ℓ += - 0.5*β*sum(abs2.(X.-μ)) + D*(0.5*N*log(β) - 0.5*N*log(2π) - 0.5*β*tr(Σ))

        # contribution of entropy 

        ℓ += D*(0.5*logdet(Σ)) 


        return ℓ - 0.5*α*sum(abs2.(Z)) # penalty on latent coordinates
    
    end
    

    objective(p) = -marginallikelihood(unpack(p)...)

    p0 = let 
        
        [randn(rg, Q*N)*0.2; randn(rg,3)*1; 0.1*randn(rg, nwts); randn(rg, N)]

    end

    @printf("Optimising %d number of parameters\n",length(p0))

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    function fg!(F, G, x)
            
        value, ∇f = Zygote.withgradient(objective, x)

        isnothing(G) || copyto!(G, ∇f[1])

        isnothing(F) || return value

        nothing

    end

    results = optimize(Optim.only_fg!(fg!), p0, LBFGS(), opt)

    Zopt, θopt, σ²opt, μopt, Λopt = unpack(results.minimizer)
 

    return Zopt, θopt, σ²opt, μopt, Λopt

end