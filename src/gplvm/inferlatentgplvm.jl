function inferlatentgplvm(ytest, R; iterations = 1000, repeats = 1) 

    @show Q = length(R[:Z][:,1]) # dimension of latent space

    idx = findall(x->~isinf(x), ytest)

    function predictiveloglikel(x)

        local μpred, Σpred = predictgplvm(reshape(x,Q,1), R)

        local μ =  μpred[idx]
        local Σ =  Σpred[idx]

        logpdf(MvNormal(μ, diagm(Σ)), ytest[idx])
        
    end

   
    objective(p) = -predictiveloglikel(p)


    function getsolution()
        
        
        # pick randonly projections of training data as starting points for optimisation
        local init = let 
            
            local randomindices = randperm(size(R[:Z],2))[1:10]
            
            local bestindex = argmin(map(i -> objective(R[:Z][:,i]), randomindices))

            R[:Z][:,bestindex]

        end

        @printf("Optimising %d number of parameters\n",length(init))
        local optf = Optimization.OptimizationFunction((u,_)->objective(u), Optimization.AutoForwardDiff())
        local prob = Optimization.OptimizationProblem(optf, init)
        Optimization.solve(prob, NelderMead(), maxiters=iterations, callback = callback)

    end

    solutions = [getsolution() for _ in 1:repeats]

    bestindex = argmin([s.objective for s in solutions])

    return solutions[bestindex].u

end