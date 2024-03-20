infertestlatent_photo(U,B,S,R; iterations = 1000, repeats = 1) = infertestlatent_photo(U, B, S; R..., iterations = iterations, repeats = repeats)

function infertestlatent_photo(U, B, S; μ = μ, Σ = Σ, K = K, η = η, Λroot = Λroot, net = net, w = w,
                             α = α, b = b, 𝛃 = 𝛃, Z = Z, θ = θ, JITTER = JITTER, rg = rg, iterations = iterations, repeats = repeats)

    # work out and verify dimensions
    D, N = size(μ); @assert(N == size(Z, 2) == size(Λroot, 1) == size(Λroot, 2))

    Q = size(Z, 1)

    J, N₊ = size(U); @assert(size(B,1) == J); @assert(size(B, 2) == D); @assert(size(S) == size(U))


    # pre-calculate
    
    inv_K_plus_Λ⁻¹ = aux_invert_K_plus_Λ⁻¹(K=K, Λroot=Λroot)

    inv_K_mul_μ = K\μ'


    # convenient, shorter name

    unpack(p) = unpack_inferlatent_gplvmplus(p ; Q = Q, N₊ = N₊, w = w, net = net)


    #--------------------------------------------------
    function objective(Z₊, ν, Lroot)
    #--------------------------------------------------

        # return partial log-likelihood composed of sum of log-prior contribution, entropy, penalty on latent coordinates

        local ℓ, A = partial_objective(Z₊, ν, Lroot; Z = Z, θ = θ, JITTER = JITTER, η = η, D = D, inv_K_plus_Λ⁻¹ = inv_K_plus_Λ⁻¹, inv_K_mul_μ = inv_K_mul_μ)

        # log-likelihood contribution

        for t in 1:N₊, j in 1:J
            
            # local aux = zero(eltype(ν))
            # for d in 1:D
            #     aux += B[j,d] * E(a = α, μ = ν[d,t], σ = sqrt(A[t,t]),b = b)
            # end
            
            # line below implements commented-out code above
            auxE = sum( B[j,:] .* exp.(α*ν[:,t]   .+     α^2*A[t,t] / 2 .+  b) )

            ℓ += logpdf(Normal(auxE, S[j,t]), U[j,t])

            # local aux_tr = zero(eltype(ν))
            # for d in 1:D
            #     aux_tr += B[j,d]^2 * V(a = α, μ = ν[d,t], σ = sqrt(A[t,t]),b = b)
            # end

            # line below implements commented-out code above. Use property V[X] = E[X²] - (E[X])²
            aux_V = sum( B[j,:].^2 .* (exp.(2*α*ν[:,t] .+ (2*α)^2*A[t,t] / 2 .+ 2b) .- (exp.(α*ν[:,t]   .+     α^2*A[t,t] / 2 .+  b)).^2) )

        
            ℓ += -(0.5 / S[j,t]^2) * aux_V

        end

        return ℓ
        
    end


    #-----------------------------------------------------------------
    # initialise parameters by picking random coordinate and random
    # values for the sqrt diagonal parametrising the posterior cov
    #-----------------------------------------------------------------

    @assert(N₊ == 1)

    function p0()

        local luckyindex = ceil(Int, rand(rg)*(size(Z,2))) 
        
        [Z[:,luckyindex]; randn(rg, N₊)]

    end
    
    #-----------------------------------------------------------------
    # define options, loss and gradient to be passed to Optim.optimize
    #-----------------------------------------------------------------

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    objective(p) = -objective(unpack(p)...)

    fg! = getfg!(objective)
    
    
    #-----------------------------------------------------------------
    # Carry out actual optimisation and obtain optimised parameters
    #-----------------------------------------------------------------

    @printf("Optimising %d number of parameters\n",length(p0()))

    solutions = [optimize(Optim.only_fg!(fg!), p0(), ConjugateGradient(), opt) for _ in 1:repeats] # alphaguess = InitialQuadratic(α0=1e-8)

    bestindex = argmin([s.minimum for s in solutions])

    Zopt, _νopt, _Lroot = unpack(solutions[bestindex].minimizer)
    
    return Zopt
    
end