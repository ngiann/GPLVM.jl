#-------------------------------------------------#
# Expectations of exponential function wrt Normal #
#-------------------------------------------------#

# ∫ exp(x) N(x|μ,σ) dx

E_(μ, σ)  = exp(μ + σ^2 / 2)

# ∫ exp(x + b) N(x|μ,σ) dx

E_(μ, σ, b)  = exp(μ + σ^2 / 2 + b)

# ∫ exp(a*x) N(x|μ,σ) dx

E_(a, μ, σ, b) = E_(a*μ, a*σ, b)

# The method below will be the one used

E(;a = 1.0, μ = μ, σ = σ, b = 0.0) = E_(a, μ, σ, b)

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
# Test with block below
# let
#     a, μ, σ, b = 1.7, -1.2, 0.5, 1.0
#     p = Normal(μ, σ)
#     quadgk(x -> pdf(p,x) * exp(a*x+b),-30.0,30.0)[1], PositiveGP.E(α=a, μ=μ, σ=σ, b=b)
#  end
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


#---------------------------------------------------------#
# Expectations of squared exponential function wrt Normal #
#---------------------------------------------------------#

B_(μ, σ) = E_(2μ, 2σ)

B_(μ, σ, b) = E_(2*μ, 2*σ, 2*b)

B_(α, μ, σ, b) = E_(2*α*μ, 2*α*σ, 2b)

B(;a=a, μ=μ, σ=σ, b=b) = E_(2*a*μ, 2*a*σ, 2b)

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# Test with block below
#     let
#     a, μ, σ, b = 1.7, 1.2, 0.85, 1.0
#     p = Normal(μ, σ)
#     quadgk(x -> pdf(p,x) * exp(a*x+b)^2,-30.0,30.0)[1], PositiveGP.E²(a=a, μ=μ,σ= σ,b= b)
# end
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


#---------------------------------------------#
# Variance of exponential function wrt Normal #
#---------------------------------------------#

V_(μ, σ) = B_(μ, σ) - E_(μ, σ)^2

V_(α, μ, σ) = B_(α, μ, σ) - E_(α, μ, σ)^2

V_(α, μ, σ, b) = B_(α, μ, σ, b) - E_(α, μ, σ, b)^2

V(;a=a, μ=μ, σ=σ, b=b) = V_(a, μ, σ, b)

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# Test with block below
# let
#     a, μ, σ, b = 1.9, -1.2, 0.95, 3.0
#     p = Normal(μ, σ); m = PositiveGP.E(a=a, μ=μ, σ=σ, b=b)
#     quadgk(x -> pdf(p,x) * (exp(a*x+b)-m)^2,-23.0,23.0)[1], PositiveGP.V(a, μ, σ, b)
# end
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *



#---------------------------------------------#
# Convenience methods involving two normals   #
# ∫∫N(x|μ₁,σ₁)N(y|μ₂,σ₂) exp(x+y) dx dy       #
#---------------------------------------------#

Exy_(μ₁, σ₁, μ₂, σ₂) = exp(μ₁ + σ₁^2 / 2 + μ₂ + σ₂^2 / 2)

Exy_(μ₁, σ₁, μ₂, σ₂, b) = exp(μ₁ + σ₁^2 / 2 + μ₂ + σ₂^2 / 2 + b)

Exy_(α₁, μ₁, σ₁, α₂, μ₂, σ₂, b) = Exy_(α₁*μ₁, α₁*σ₁, α₂*μ₂, α₂*σ₂, b)

Exy(;a₁=a₁, μ₁=μ₁, σ₁=σ₁, a₂=a₂, μ₂=μ₂, σ₂=σ₂, b=b) = Exy_(a₁, μ₁, σ₁, a₂, μ₂, σ₂, b)

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
# Test with block below
# let
#     b = 0.76
#     a₁, μ₁, σ₁ = 1, -1.2, 0.50
#     a₂, μ₂, σ₂ = 1, -0.9, 0.75
#     p₁ = Normal(μ₁, σ₁)
#     p₂ = Normal(μ₂, σ₂)
#     hcubature(xy -> pdf(p₁,xy[1]) * pdf(p₂,xy[2]) * exp(a₁*xy[1] + a₂*xy[2] + b),-30.0*ones(2),30.0*ones(2))[1], PositiveGP.Exy(a₁=a₁, μ₁=μ₁, σ₁=σ₁, a₂=a₂, μ₂=μ₂, σ₂=σ₂, b=b)
# end
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *




#---------------------------------------------#
# ∫ N(x|μ, σ) log N(y | exp(a⋅x + b), β⁻¹) dx #
#---------------------------------------------#

Elognormal(;y=y, a=a, μ=μ, b=b, σ=σ, β=β) = -0.5*log(2π) + 0.5*log(β) - 0.5*β*abs2(y - E(a = a, μ = μ, σ = σ, b = b)) - β/2 * V(a = a, μ = μ, σ = σ, b = b)

# Elognormal(y, a, μ, b, σ, β) = logpdf(Normal(E(a = a, μ = μ, σ = σ, b = b), sqrt(1/β)), y) - β/2 * V(a = a, μ = μ, σ = σ, b = b)

# let

#     y, a, μ, σ, b, β = 0.4, 1.9, -1.2, 0.95, 3.0, 1/1.3
    
#     p = Normal(μ, σ)

#     quadgk(x -> pdf(p, x) * logpdf(Normal(exp(a*x+b), sqrt(1/β)), y),-23.0,23.0)[1], PositiveGP.Elognormal(;y=y, a=a, μ=μ, b=b, σ=σ, β=β)

# end


Elognormal_barrier(y::Missing; a=a, μ=μ, b=b, σ=σ, β=β) = 0.0

Elognormal_barrier(y; a=a, μ=μ, b=b, σ=σ, β=β) = Elognormal(;y=y, a=a, μ=μ, b=b, σ=σ, β=β)