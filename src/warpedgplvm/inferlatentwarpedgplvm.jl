"""
inferlatentwarpedgplvm(Y₊, R; iterations = 1000, repeats = 10, seed = 1) 

    Input Y₊ must be of dimensions D×N₊
"""
function inferlatentwarpedgplvm(Y₊, R; iterations = 1000, repeats = 10, seed = 1) 

    Q  = length(R[:Z][:,1]) # dimension of latent space
    
    D, N₊ = size(Y₊)

    @printf("Inferring latent coordinates for %d number of %d-dimensional data items.\n", N₊, D)
    
    warp = R[:warp]

    rg = MersenneTwister(seed)

    idx = [findall(x->~isinf(x), Y₊[d,:]) for d in 1:D] # this needs to be changed when passing data with N₊>1 ❗❗❗❗


    #-----------------------------------------
    function unpack(z)
    #-----------------------------------------

        reshape(z, Q, N₊)

    end


    #-----------------------------------------
    function loss(z)
    #-----------------------------------------

        @assert(size(z) == (Q, N₊))

        local ℓ = zero(eltype(z))

        local μpred, Σpred = predictwarpedgplvm(z, R) 

        @assert(size(μpred,2) == D)
        @assert(size(Σpred[1]) == (N₊, N₊))

        for d in 1:D

            if isempty(idx[d])
                continue
            end

            local yd = warp.(Y₊[d,idx[d]]) # check marginallikelihood method in warpedgplvm.jl

            U = cholesky(Symmetric(Σpred[d][idx[d],idx[d]])).L

            ℓ += -0.5*sum(abs2.(U\(μpred[idx[d],d] - yd))) - 0.5*2*sum(log.(diag(U))) - 0.5*length(yd)*log(2π)
          
        end

        return ℓ

    end

    #-----------------------------------------
    # setup and run optimiser
    #-----------------------------------------

    opt = Optim.Options(show_trace = true, show_every = 1, iterations = iterations)

    objective(z) = -loss(unpack(z))

    
    function getsolution()
        
        # pick randonly projections of training data as starting points for optimisation
        local init = let 
    
            local randomindices = randperm(rg, size(R[:Z],2))[1:10]
            
            local bestindex = argmin(map(i -> objective(R[:Z][:,i]), randomindices))

            R[:Z][:,bestindex]

        end
        
        optf = Optimization.OptimizationFunction((u,_)->objective(u), Optimization.AutoZygote())
        prob = Optimization.OptimizationProblem(optf, init)
        Optimization.solve(prob, ConjugateGradient(), maxiters=iterations, callback = callback)

    end


    solutions = [getsolution() for _ in 1:repeats]

    bestindex = argmin([s.objective for s in solutions])

    return unpack(solutions[bestindex].u)
    
end