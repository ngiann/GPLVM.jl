function inferlatentgplvmvar_photo(U, B, S, R; iterations = 1000, repeats = 10, seed = 1) 

    @show Q = length(R[:Z][:,1]) # dimension of latent space
    @show T = size(U,1)
    @show J, D = size(B)
    @assert(size(U,2) == size(B,1))
    @assert(size(U) == size(S))

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
    # β      = R[:𝛃][1]

    rg = MersenneTwister(seed)


    # pre-calculate
    
    inv_K_plus_Λ⁻¹ = aux_invert_K_plus_Λ⁻¹(K=K, Λroot=Λroot)

    inv_K_mul_μ = K\μ'

    # use the same unpacking function like GPLVM₊

    unpack(p) = unpack_inferlatent_gplvmplus(p ; Q = Q, N₊ = T, w = w, net = net)

    function loss(Z₊, ν, Lroot)

        @assert(size(ν,2) == T)
        @assert(size(ν,1) == D)

        # use same function like GPLVM₊
        # return partial log-likelihood composed of sum of log-prior contribution, entropy, penalty on latent coordinates

        local ℓ, A = partial_objective(Z₊, ν, Lroot; Z = Z, θ = θ, JITTER = JITTER, η = η, D = D, inv_K_plus_Λ⁻¹ = inv_K_plus_Λ⁻¹, inv_K_mul_μ = inv_K_mul_μ)

        # log-likelihood contribution

        for j in 1:J, t in 1:T
            
            aux = sum(B[j,:] .* (ν[:,t] .+ b))
            
            ℓ += logpdf(Normal(aux, S[t,j]), U[t,j])

        end

        for j in 1:J, t in 1:T, d in 1:D

            ℓ += - (1/(2*S[t,j]^2) * B[j,d]^2 * A[t,t])

        end

        return ℓ

      
    end

    
    opt = Optim.Options(show_trace = true, show_every = 1, iterations = iterations)

    objective(p) = -loss(unpack(p)...)

    
    function getsolution()
        
        luckyindex = ceil(Int, rand(rg) * size(R[:Z],2)) # pick a random coordinate as starting point for optimisation
     
        init = optimize(objective, [Z[:,luckyindex]; randn(rg, T)], NelderMead(), opt).minimizer

        optimize(objective, init, ConjugateGradient(), opt, autodiff=:forward)

    end


    solutions = [getsolution() for _ in 1:repeats]

    bestindex = argmin([s.minimum for s in solutions])

    
    return unpack(solutions[bestindex].minimizer)[1]

end