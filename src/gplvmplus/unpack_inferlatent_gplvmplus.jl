function unpack_inferlatent_gplvmplus(p; D = D, Q = Q, N₊ = N₊)

    local MARK = 0

    local Z₊ = reshape(p[MARK+1:MARK+Q*N₊], Q, N₊); MARK += Q*N₊
    
    local ν = p[MARK+1:MARK+D]; MARK += D

    local Lroot = Diagonal(p[MARK+1:MARK+N₊]); MARK += N₊
    
    local c = p[MARK+1]^2; MARK += 1

    @assert(MARK == length(p)) # all parameters must be used up

    return Z₊, ν, Lroot, c

end