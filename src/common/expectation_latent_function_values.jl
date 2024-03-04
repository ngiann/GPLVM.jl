function expectation_latent_function_values(;α = α, b = b, μ = μ, Σ = Σ)

    E = exp.(α*μ   .+     α^2*diag(Σ)' / 2 .+  b)

    B = exp.(2*α*μ .+ (2*α)^2*diag(Σ)' / 2 .+ 2b) 

    V = B .- E.^2 # this is V[X] = E[X²] - (E[X])² # There may be a computational gain to be had here

    return E, V

end
