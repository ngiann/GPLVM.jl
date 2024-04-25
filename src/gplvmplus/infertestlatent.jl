function inferlatent(X₊, R; iterations = 1000, repeats=1, seed = 1) 

    infertestlatent(X₊; β = R[:β], μ = R[:μ], Σ = R[:Σ], K = R[:K], η = R[:η], Λroot = R[:Λroot], net = R[:net], w = R[:w],
    α = R[:α], b = R[:b], Z = R[:Z], θ = R[:θ], JITTER = R[:JITTER], rg = R[:rg], iterations = iterations, repeats = repeats, seed = seed)

end


function infertestlatent(X₊; β = β, μ = μ, Σ = Σ, K = K, η = η, Λroot = Λroot, net = net, w = w,
                             α = α, b = b, Z = Z, θ = θ, JITTER = JITTER, rg = rg, iterations = iterations, repeats = repeats, seed = seed)

    # sort out dimensions

    D, N = size(μ); @assert(N == size(Z, 2) == size(Λroot, 1) == size(Λroot, 2))

    Q = size(Z, 1)

    N₊ = size(X₊, 2); @assert(D == size(X₊, 1))

    rg = MersenneTwister(seed)


    # pre-calculate
    
    inv_K_plus_Λ⁻¹ = aux_invert_K_plus_Λ⁻¹(K=K, Λroot=Λroot)

    inv_K_mul_μ = K\μ'


    # convenient, shorter name

    unpack(p) = unpack_inferlatent_gplvmplus(p ; D = D, Q = Q, N₊ = N₊)
    

    idx = findall(x -> ~isinf(x), X₊)


    #--------------------------------------------------
    function lowerbound(Z₊, ν, Lroot, c)
    #--------------------------------------------------


        # return partial log-likelihood composed of sum of log-prior contribution, entropy, penalty on latent coordinates

        local ℓ, A = partial_objective(Z₊, ν, Lroot; Z = Z, θ = θ, JITTER = JITTER, η = η, D = D, inv_K_plus_Λ⁻¹ = inv_K_plus_Λ⁻¹, inv_K_mul_μ = inv_K_mul_μ)

        # log-likelihood contribution

        local E, V = expectation_latent_function_values(;α = α, b = b, μ = ν, Σ = A)

        ℓ += -0.5*β*sum(abs2.((X₊[idx] - c*E[idx]))) - 1/2 * β * c * c * sum(V[idx])
       
        
        return ℓ

    end


    #-----------------------------------------------------------------
    # define options, loss and gradient to be passed to Optim.optimize
    #-----------------------------------------------------------------

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    objective(p) = -lowerbound(unpack(p)...)


    #-----------------------------------------------------------------
    # initialise parameters by starting from randomly picked latent
    # coordinate and refining it initially with robust NelderMead
    #-----------------------------------------------------------------

    @assert(N₊ == 1)


    

    function p0()

        local localopt = Optim.Options(iterations = 300, show_trace = true, show_every = 1, g_tol=1e-5)

        local luckyindex = ceil(Int, rand(rg)*(size(Z,2))) 

        local fixX = reshape((Z[:,luckyindex]), Q, N₊)
   
        function partialunpack(p)

            local Z₊, ν, Lroot, c = unpack([fixX;p])

            Z₊, ν, Lroot, c 

        end

        objpartial(p) = -lowerbound(partialunpack(p)...)
             
        local partialsol = [randn(rg, D*N₊); randn(rg, N₊); 1.0]

        local partialresult = optimize(objpartial, partialsol, ConjugateGradient(), localopt, autodiff=:forward)

        [Z[:,luckyindex]; partialresult.minimizer]

    end


    #-----------------------------------------------------------------
    # Carry out actual optimisation and obtain optimised parameters
    #-----------------------------------------------------------------

    @printf("Optimising %d number of parameters\n", size(Z,1)+N₊)

    # solutions = [optimize(objective, p0(), ConjugateGradient(), opt, autodiff=:forward) for _ in 1:repeats]
###########
    solutions = map(1:repeats) do repeatindex
        optf = Optimization.OptimizationFunction((u,_)->objective(u), Optimization.AutoZygote())
        prob = Optimization.OptimizationProblem(optf, p0())
        Optimization.solve(prob, ConjugateGradient(), maxiters=iterations, callback = callback)
    end

###########
    bestindex = argmin([s.objective for s in solutions])

    @printf("best fmin=%f\n", solutions[bestindex].objective)

    Zopt, νopt, Lroot, copt = unpack(solutions[bestindex].u)

    return Zopt, copt

end