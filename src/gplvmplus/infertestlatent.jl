# There are two cases cases:

# 1. At testing no error measurements provided: assume at training GPLVM₊ precision was optimised.

function inferlatent(X₊, R; iterations = 10, repeats=1) 
    
    @assert isa(R[:𝛃], FillArrays.AbstractFillMatrix)

    infertestlatent(X₊, Fill(R[:𝛃][1], size(X₊)); μ = R[:μ], Σ = R[:Σ], K = R[:K], η = R[:η], Λroot = R[:Λroot], net = R[:net], w = R[:w],
    α = R[:α], b = R[:b], Z = R[:Z], θ = R[:θ], JITTER = R[:JITTER], rg = R[:rg], iterations = iterations, repeats = repeats)

end

# 2. At testing error measuments are provided.

inferlatent(X₊, 𝛔, R; iterations = iterations, repeats = repeats) = infertestlatent(X₊, inverterrors(𝛔);  μ = R[:μ], Σ = R[:Σ], K = R[:K], η = R[:η], Λroot = R[:Λroot], net = R[:net], w = R[:w],
α = R[:α], b = R[:b], Z = R[:Z], θ = R[:θ], JITTER = R[:JITTER], rg = R[:rg], iterations = iterations, repeats = repeats)



function infertestlatent(X₊, 𝛃; μ = μ, Σ = Σ, K = K, η = η, Λroot = Λroot, net = net, w = w,
                             α = α, b = b, Z = Z, θ = θ, JITTER = JITTER, rg = rg, iterations = iterations, repeats = repeats)

    # sort out dimensions

    D, N = size(μ); @assert(N == size(Z, 2) == size(Λroot, 1) == size(Λroot, 2))

    Q = size(Z, 1)

    N₊ = size(X₊, 2); @assert(D == size(X₊, 1)); @assert(size(X₊) == size(𝛃))


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

        local E, V = expectation_latent_function_values(;α = α, b = b, μ = ν, Σ = A)

        ℓ += -0.5*D*N₊*log(2π) + 0.5*sum(log.(𝛃))  -0.5*sum(𝛃 .* abs2.(myskip.(X₊ .- E))) - 1/2 * sum(𝛃 .* V)

        return ℓ

    end


    # initialise parameters randomly
    @assert(N₊ == 1)

    function p0()

        local luckyindex = ceil(Int, rand(rg)*(size(Z,2))) 
        
        [Z[:,luckyindex]; randn(rg, N₊)]

    end


    #-----------------------------------------------------------------
    # define options, loss and gradient to be passed to Optim.optimize
    #-----------------------------------------------------------------

    opt = Optim.Options(iterations = iterations, show_trace = false, show_every = 1)

    objective(p) = -objective(unpack(p)...)

    fg! = getfg!(objective)


    #-----------------------------------------------------------------
    # Carry out actual optimisation and obtain optimised parameters
    #-----------------------------------------------------------------

    @printf("Optimising %d number of parameters\n",length(p0()))

    solutions = [optimize(Optim.only_fg!(fg!), p0(), ConjugateGradient(), opt) for _ in 1:repeats]

    bestindex = argmin([s.minimizer for s in solutions])

    Zopt, νopt, Lroot = unpack(solutions[bestindex].minimizer)
   
    return Zopt

end