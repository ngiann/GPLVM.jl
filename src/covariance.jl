function covariance(D², θ)

    θ[1] .* exp.(-D² .* θ[2])

end