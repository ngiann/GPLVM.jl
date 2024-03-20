function unpack_gplvm(p, D, N, Q)

    @assert(length(p) == Q*N + 2 + 1 + 1)

    MARK = 0

    Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

    θ = exp.(p[MARK+1:MARK+2]); MARK += 2

    σ² = exp(p[MARK+1]); MARK += 1

    b = p[MARK+1]; MARK += 1

    @assert(MARK == length(p))

    return Z, θ, σ², b

end