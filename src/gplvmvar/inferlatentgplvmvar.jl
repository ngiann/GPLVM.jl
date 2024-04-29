function inferlatentgplvmvar(X, R; iterations = 1000, repeats = 10, seed = 1) 

    @show Q  = length(R[:Z][:,1]) # dimension of latent space
    @show N₊ = 1
    @show D  = length(X)
    
    # assign relevant quantities

    K      = R[:K]
    Λroot  = R[:Λroot]
    b      = R[:b]
    μ      = R[:μ]
    # Σ      = R[:Σ]
    η      = R[:η]
    net    = R[:net]
    w      = R[:w]
    Z      = R[:Z]
    θ      = R[:θ]
    JITTER = R[:JITTER]
    β = R[:β]
   

    rg = MersenneTwister(seed)


    notinf(x) = ~isinf(x)

    idx = findall(notinf.(X))

    @show length(idx)/length(X)
    
    # pre-calculate
    
    inv_K_plus_Λ⁻¹ = aux_invert_K_plus_Λ⁻¹(K=K, Λroot=Λroot)

    inv_K_mul_μ = K\μ'

    # use the same unpacking function like GPLVM₊

    unpack(p) = unpack_inferlatent_gplvmvar(p ; D = D, Q = Q, N₊ = N₊)

    function loss(Z₊, ν, Lroot)
        
        # use same function like GPLVM₊
        # return partial log-likelihood composed of sum of log-prior contribution, entropy, penalty on latent coordinates

        local ℓ, A = partial_objective(Z₊, ν, Lroot; Z = Z, θ = θ, JITTER = JITTER, η = η, D = D, inv_K_plus_Λ⁻¹ = inv_K_plus_Λ⁻¹, inv_K_mul_μ = inv_K_mul_μ)

        # log-likelihood contribution

        # @views ℓ += - 0.5*sum(β[idx] .* abs2.(((X[idx].-ν[idx].-b)))) + 0.5*sum((log.(β))) - 0.5*sum(β*diag(A))
        @views ℓ += - 0.5*sum(β .* abs2.(((X[idx].-ν[idx].-b)))) - 0.5*sum(β*diag(A))

        return ℓ

    end

    
    opt = Optim.Options(show_trace = true, show_every = 10, iterations = iterations)

    objective(p) = -loss(unpack(p)...)

    
    function getsolution()
        
        luckyindex = ceil(Int, rand(rg) * size(R[:Z],2)) # pick a random coordinate as starting point for optimisation
     
        init = [Z[:,luckyindex]; randn(rg, D*N₊); randn(rg, N₊)]

        # optimize(objective, init, ConjugateGradient(), opt, autodiff=:forward)

        optf = Optimization.OptimizationFunction((u,_)->objective(u), Optimization.AutoZygote())
        prob = Optimization.OptimizationProblem(optf, init)
        sol  = Optimization.solve(prob, ConjugateGradient(), maxiters=iterations, callback = callback)
    
    end


    solutions = [getsolution() for _ in 1:repeats]

    bestindex = argmin([s.objective for s in solutions])

    
    return unpack(solutions[bestindex].u)[1]

end