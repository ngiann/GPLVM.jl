function woodbury(;K = K, Λ½ = Λ½) 
    
    # sources:
    # see 145 in matrix coobook where we set A⁻¹=K, B=I, C = Λ½
    # see (A.9) in GPML 
    # The application of the woodbury identity is recommended in GPML, see (3.26) and (3.27).

    KΛ½ = K*Λ½

    U = cholesky(Symmetric(I +  Λ½*KΛ½)).L

    A = U\KΛ½'

    Symmetric(K - A'A)

end