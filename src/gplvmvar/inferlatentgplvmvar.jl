function inferlatentgplvmvar(X₊, R; iterations = 1000, repeats = 10) 

    @show Q  = length(R[:Z][:,1]) # dimension of latent space
    @show N₊ = 1
    @show D  = length(X₊)
    
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
    β      = R[:𝛃][1]

    rg = MersenneTwister(1)


    countObs = count(x->~ismissing(x), X₊)
    
    # pre-calculate
    
    inv_K_plus_Λ⁻¹ = aux_invert_K_plus_Λ⁻¹(K=K, Λroot=Λroot)

    inv_K_mul_μ = K\μ'

    # use the same unpacking function like GPLVM₊

    unpack(p) = unpack_inferlatent_gplvmplus(p ; Q = Q, N₊ = N₊, w = w, net = net)

    function loss(Z₊, ν, Lroot)
        
        # use same function like GPLVM₊
        # return partial log-likelihood composed of sum of log-prior contribution, entropy, penalty on latent coordinates

        local ℓ, A = partial_objective(Z₊, ν, Lroot; Z = Z, θ = θ, JITTER = JITTER, η = η, D = D, inv_K_plus_Λ⁻¹ = inv_K_plus_Λ⁻¹, inv_K_mul_μ = inv_K_mul_μ)

        # log-likelihood contribution

        ℓ += - 0.5*β*sum(abs2.(myskip.((X₊.-ν.-b)))) + 0.5*countObs*log(β) - 0.5*countObs*log(2π) - 0.5*β*D*tr(A)

        return ℓ

    end

    
    opt = Optim.Options(show_trace = true, show_every = 1, iterations = iterations)

    objective(p) = -loss(unpack(p)...)

    
    function getsolution()
        
        luckyindex = ceil(Int, rand(rg) * size(R[:Z],2)) # pick a random coordinate as starting point for optimisation
     
        init = optimize(objective, [Z[:,luckyindex]; randn(rg, N₊)], NelderMead(), opt).minimizer

        optimize(objective, init, LBFGS(), opt, autodiff=:forward)

    end


    solutions = [getsolution() for _ in 1:repeats]

    bestindex = argmin([s.minimum for s in solutions])

    
    return unpack(solutions[bestindex].minimizer)[1]

end