function inferlatentgplvmvar(ytest, R; iterations = 1000, repeats = 10) 

    Q = length(R[:Z][:,1]) # dimension of latent space

    CountObs = length(ytest) - count(ismissing, ytest)
    

    function loss(x)

        μpred, Σpred = predictgplvmvar(reshape(x,Q,1), R)

        return -0.5*CountObs*log(2π) - 0.5*(sum(myskip.(abs2.(((ytest - vec(μpred)))))))/only(Σpred) - 0.5*CountObs*log(only(Σpred))

        # # code below implements line above - keep for verification

        # ℓ = zero(eltype(x))
        # for (m,y) in zip(μpred,ytest)
        #     if ~ismissing(y)
        #         ℓ += logpdf(Normal(m, sqrt(only(Σpred))), y)
        #     end  
        # end
        # ℓ

    end

    
    opt = Optim.Options(show_trace = true, show_every = 1, iterations = iterations)

    objective(p) = -loss(p)

    
    function getsolution()
        
        luckyindex = ceil(Int, rand() * size(R[:Z],2)) # pick a random coordinate as starting point for optimisation
     
        init = optimize(objective, R[:Z][:,luckyindex], NelderMead(), opt).minimizer

        optimize(objective, init, LBFGS(), opt, autodiff=:forward)

    end


    solutions = [getsolution() for _ in 1:repeats]

    bestindex = argmin([s.minimum for s in solutions])

    return solutions[bestindex].minimizer

end