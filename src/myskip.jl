myskip(x::Missing) = 0.0 # maybe we should make this zero(eltype(T))

myskip(x) = x