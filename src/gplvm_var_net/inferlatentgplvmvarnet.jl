function inferlatentgplvmvarnet(ytest, R; iterations = 1000, repeats = 10) 


    Q = length(R[:Z][:,1]) # dimension of latent space


    function loss(x)

        # for the purpose of inferring the latent coordinate
        # it suffices to minimise dicrepancy from mean prediction

        local μpred, = predictgplvmvarnet(reshape(x,Q,1), R)

        -0.5*sum(abs2.(vec(μpred) - vec(ytest)))

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

        gradloss!(s,p) = copyto!(s, Zygote.gradient(x->-loss(x),p)[1])

        optimize(Optim.only_fg!(fg!), R[:Z][:,luckyindex], ConjugateGradient(), opt)

    end


    solutions = [getsolution() for _ in 1:repeats]

    bestindex = argmin([s.minimum for s in solutions])

    return solutions[bestindex].minimizer

end