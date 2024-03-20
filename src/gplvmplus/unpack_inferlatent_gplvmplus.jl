function unpack_inferlatent_gplvmplus(p; Q = Q, N₊ = N₊, w = w, net = net)

    local MARK = 0

    local Z₊ = reshape(p[MARK+1:MARK+Q*N₊], Q, N₊); MARK += Q*N₊
    
    local Lroot = Diagonal(p[MARK+1:MARK+N₊]); MARK += N₊
    
    @assert(MARK == length(p)) # all parameters must be used up

    local ν  = net(w, Z₊)

    return Z₊, ν, Lroot

end