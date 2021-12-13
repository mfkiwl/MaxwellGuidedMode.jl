export DispersiveMode
export DispersiveField
export calc_dispersion

const DispersiveMode{K,Kₑ,Kₘ,K₊₁,VR<:AbstractRange{<:Real}} = ParametrizedMode{K,Kₑ,Kₘ,1,K₊₁,NamedTuple{(:ω,),Tuple{VR}}}

DispersiveMode(ω::AbsVecReal, mdl::Model) = (θ = (ω=ω,); ParametrizedMode(θ,mdl))

struct DispersiveField{K}
    Fᵣ_fun::Array{Spline1D,K}
    Fᵢ_fun::Array{Spline1D,K}
end

function DispersiveField(Ω::AbstractRange{<:Real},
                         F::AbsArrNumber{K₊₁}  # F[Ω,x,y,z]
                         ) where {K₊₁}
    size(F,1)==length(Ω) || @error "size(F,1) = $(size(F,1)) and length(Ω) = $(length(Ω)) should be same."

    K = K₊₁ - 1
    dim_shp = SVec(ntuple(k->k+1, Val(K)))  # 2:K+1
    sz_shp = size(F)[dim_shp]
    CI = CartesianIndices(sz_shp)

    bc = "extrapolate"
    Fᵣ_fun = Array{Spline1D}(undef, sz_shp)
    Fᵢ_fun = Array{Spline1D}(undef, sz_shp)
    for ci = CI
        Fᵣ_fun[ci] = Spline1D(Ω, real.(@view(F[:,ci])); bc)
        Fᵢ_fun[ci] = Spline1D(Ω, imag.(@view(F[:,ci])); bc)
    end

    return DispersiveField(Fᵣ_fun, Fᵢ_fun)
end

Base.size(df::DispersiveField) = size(df.Fᵣ_fun)

function (df::DispersiveField)(Ω::Real)
    sz_shp = size(df)
    F = ArrComplexF(undef, sz_shp)

    CI = CartesianIndices(F)
    for ci = CI
        F[ci] = df.Fᵣ_fun[ci](Ω) + im * df.Fᵢ_fun[ci](Ω)
    end

    return F
end

# Calculate quantities related to group velocity dispersion (GVD).  These quantities are
# usually calculated around a specific frequency ω₀, which is provided to the function.
#
# This often generates the final results for publication, so the outputs are described in
# the SI units.
function calc_dispersion(dm::DispersiveMode{K,Kₑ,Kₘ}, ω₀::Real, unit::MaxwellUnit) where {K,Kₑ,Kₘ}
    ω = dm.θ.ω
    ω[1] ≤ ω₀ ≤ ω[end] || @error "ω₀ = $ω₀ should be within range of dm.θ.ω."

    # Convert the units of dm's quantities to the SI units.
    ω₀ *= unit.ω
    ω *= unit.ω  # ω is newly allocated because *= instead of .*= is used
    ωmin, ωmax, Nω = ω[1], ω[end], length(ω)

    β = real.(dm.β) ./ unit.L  # β is newly allocated
    neff = c₀ .* (β ./ ω)  # ω = c β = (c₀ / neff) β

    # dm's E and H are normalized to have unit longitudinal energy flux, but the energy flux
    # is in the units of unit.P, which is not 1.  Once E, H, and length in transverse
    # dimension are multiplied by unit.E, unit.H, and unit.L, energy flux is multiplied by
    # unit.P.  In order to normalize E and H in the SI units whose longitudinal energy flux
    # is unit.P, we need to normalize both fields by the same factor, which is √unit.P.
    E = dm.E .* (unit.E / √unit.P)  # E is newly allocated
    H = dm.H .* (unit.H / √unit.P)  # H is newly allocated

    # Prepare interpolation.
    Ωmin, Ωmax = ωmin-ω₀, ωmax-ω₀
    Ω = range(Ωmin, Ωmax, length=Nω)  # enforce uniform sampling
    bc = "extrapolate"

    # Construct the interpolators for neff and β.
    neff_fun = Spline1D(Ω, neff; bc)
    β_fun = Spline1D(Ω, β; bc)

    # Construct the interpolators for phase mismatch ∆β.
    β₀ = β_fun(0.0)
    β₁₀ = derivative(β_fun, 0.0)
    ∆β = β .- β₀ .- β₁₀ .* Ω
    ∆β_fun = Spline1D(Ω, ∆β; bc)

    # Construct the interpolator for group velocity vg.
    # Another option is to define a function that evaluates the derivatives on the fly as
    # vg_fun(Ω) = 1 / derivative(β_fun, Ω), but this turns out to be very slow.  So we
    # evaluate the derivatives at the sampling points and construct a new interpolator for
    # the derivatives.
    β₁ = derivative(β_fun, Ω)
    β₁_fun = Spline1D(Ω, β₁; bc)

    vg = 1 ./ β₁
    vg_fun = Spline1D(Ω, vg; bc)

    # Construct the interpolator for GVD parameter.
    β₂ = derivative(β_fun, Ω, nu=2)  # nu: order of derivative
    β₂_fun = Spline1D(Ω, β₂; bc)

    D = -ω.^2 .* β₂ ./ 2πc₀  # D = -(ω²/2πc₀) β₂
    D_fun = Spline1D(Ω, D; bc)

    # Construct the interpolator for E and H.
    dim_shp = SVec(ntuple(k->k+1, Val(K)))  # 2:K+1
    sz_shp = size(E[1])[dim_shp]
    CI = CartesianIndices(sz_shp)

    E_fun = ntuple(k->DispersiveField(Ω,E[k]), Val(Kₑ))
    H_fun = ntuple(k->DispersiveField(Ω,H[k]), Val(Kₘ))

    return (ω₀=ω₀, ω=ω, Ω=Ω, neff=neff_fun, β=β_fun, β₁=β₁_fun, β₂=β₂_fun, ∆β=∆β_fun, vg=vg_fun, D=D_fun, E=E_fun, H=H_fun)
end
