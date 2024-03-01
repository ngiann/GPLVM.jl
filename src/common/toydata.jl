function toydata(;σ=0.01, N=101, seed = 1)

    rg = MersenneTwister(seed)

    θ = collect(LinRange(0,2π, N))[1:N-1]

    X = Matrix(1*[sin.(θ) cos.(θ)]') .+ [3.0; 3.0]

    A = rand(rg, 5, 2)
    
    Y = A*X

    S = ones(size(Y))*σ

    return Y + randn(rg, size(Y)) .* S, S

end