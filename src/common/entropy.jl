function entropy(Σ)

    N = size(Σ, 1)

    0.5*N*log(2*π*ℯ) + 0.5*logabsdet(Σ)[1] 

end