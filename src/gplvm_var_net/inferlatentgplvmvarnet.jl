function inferlatentgplvmvarnet(ytest, R; iterations = 1000, repeats = 10) 

    Q = length(R[:Z][:,1]) # dimension of latent space

    CountObs = length(ytest) - count(ismissing, ytest)

    function loss(x)

        local μpred, Σpred = predictgplvmvarnet(reshape(x,Q,1), R)

        -0.5*CountObs*log(2π) - 0.5*myskip.(sum(abs2.(((ytest - vec(μpred))))))/only(Σpred) - 0.5*CountObs*log(only(Σpred))

        # code below implements line above - keep for verification

        # ℓ = zero(eltype(x))
        # for (m,y) in zip(μpred,ytest)
        #     if ~ismissing(y)
        #         ℓ += logpdf(Normal(m, sqrt(only(Σpred))), y)
        #     end  
        # end
        # ℓ
        
    end

    
    opt = Optim.Options(show_trace = false, show_every = 1, iterations = iterations)

    objective(p) = -loss(p)

    function fg!(F, G, x)

        value, ∇f = Zygote.withgradient(objective,x)

        isnothing(G) || copyto!(G, ∇f[1])

        isnothing(F) || return value

        nothing

    end


    function getsolution()
        
        luckyindex = ceil(Int, rand() * size(R[:Z],2)) # pick a random coordinate as starting point for optimisation
     
        init = optimize(objective, R[:Z][:,luckyindex], NelderMead(), opt).minimizer

        optimize(objective, init, ConjugateGradient(), opt, autodiff=:forward)

    end


    solutions = [getsolution() for _ in 1:repeats]

    bestindex = argmin([s.minimum for s in solutions])

    return solutions[bestindex].minimizer

end