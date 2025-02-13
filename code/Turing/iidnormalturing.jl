using Turing, Plots, LaTeXStrings

ScaledInverseChiSq(ν,τ²) = InverseGamma(ν/2,ν*τ²/2) # Scaled Inv-χ² distribution

# Setting up the Turing model:
@model function iidnormal(x, μ₀, κ₀, ν₀, σ²₀)
    σ² ~ ScaledInverseChiSq(ν₀, σ²₀)
    θ ~ Normal(μ₀, √(σ²/κ₀))  # prior
    n = length(x)  # number of observations
    for i in 1:n
        x[i] ~ Normal(θ, √σ²) # model
    end
end

# Set up the observed data
x = [15.77,20.5,8.26,14.37,21.09]

# Set up the prior
μ₀ = 20; κ₀ = 1; ν₀ = 5; σ²₀ = 5^2

# Settings of the Hamiltonian Monte Carlo (HMC) sampler.
α = 0.8
postdraws = sample(iidnormal(x, μ₀, κ₀, ν₀, σ²₀), NUTS(α), 10000, discard_initial = 1000)

p1 = histogram(postdraws.value[:,2], yaxis = false, title = L"\mu")
p2 = histogram(sqrt.(postdraws.value[:,1]), yaxis = false, title = L"\sigma")
plot(p1, p2, layout = (1,2), size = (600,300))



