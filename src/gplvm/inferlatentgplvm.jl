function inferlatentgplvm(ytest, R; iterations = 100, repeats = 1) 

    @show Q = length(R[:Z][:,1]) # dimension of latent space

    idx = findall(x->~isinf(x), ytest)

    function loss(x)

        local μpred, Σpred = predictgplvm(reshape(x,Q,1), R)

        local μ =  μpred[idx]
        local Σ =  Σpred[idx]

        logpdf(MvNormal(μ, diagm(Σ)), ytest[idx])
        
    end

    
    opt = Optim.Options(show_trace = true, show_every = 1, iterations = iterations)

    objective(p) = -loss(p)

    # function fg!(F, G, x)

    #     value, ∇f = Zygote.withgradient(objective,x)

    #     isnothing(G) || copyto!(G, ∇f[1])

    #     isnothing(F) || return value

    #     nothing

    # end

    function getsolution()
        
        luckyindex = ceil(Int, rand() * size(R[:Z],2)) # pick a random coordinate as starting point for optimisation
     
        bnd = (minimum(vec(R[:Z])), maximum(vec(R[:Z])))

        init = best_candidate(bboptimize(objective; Method=:random_search, SearchRange = bnd, NumDimensions = Q, MaxFuncEvals = 100))
 
        optimize(objective, init, NelderMead(), opt)

        # optimize(objective, init, NelderMead(), opt)

    end


    solutions = [getsolution() for _ in 1:repeats]

    bestindex = argmin([s.minimum for s in solutions])

    return solutions[bestindex].minimizer

end