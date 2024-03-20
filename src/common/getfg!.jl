function getfg!(objective)

    function fg!(F, G, x)
                
            value, ∇f = Zygote.withgradient(objective,x)

            isnothing(G) || copyto!(G, ∇f[1])

            isnothing(F) || return value

            nothing

    end

end