# Alias of Model{K,Kₑ,Kₘ, Kₑₜ,Kₘₜ, Kₑₗ,Kₘₗ, K², K₊₂,AK₊₂} for the TE waveguide mode
# mode equations for 1D cross section.
#
# The type parameter AK₊₂ specify device-specific arrays types (e.g., CuArray) and is
# user-defined in the constructor.
#
# Note that the TE mode has (Ey, Hx, Hz) as nonzero components; this is unlike TE Maxwell's
# equations that have two E-field components and one H-field component.

const ModelTE{AK₊₂} = Model{1,1,2, 1,1, 0,1, 1, 3,AK₊₂}

# Convenience constructor
function ModelTE(grid::Grid; Atype::Type=Array)
    cmpₛ = SInt(1)  # shapes in transverse dimension (x-axis)

    cmpₑₜ = SInt(2)  # transverse E-field (y-component)
    cmpₘₜ = SInt(1)  # transverse H-field (x-component)

    cmpₑₗ = SInt()  # longitudinal E-field (no component)
    cmpₘₗ = SInt(3)  # longitudinal H-field (z-component)

    iseₜ˔shp = true
    ishₜ˔shp = false

    return ModelTE{Atype{ComplexF,3}}(;grid, cmpₛ, cmpₑₜ, cmpₘₜ, cmpₑₗ, cmpₘₗ, iseₜ˔shp, ishₜ˔shp)
end

# Assign the material parameters on the grid and smooth them.
function calc_matparams!(mdl::ModelTE)
    # Take necessary fields of mdl.
    grid = mdl.grid

    isbloch = grid.isbloch
    N = grid.N
    l = grid.l
    σ = grid.σ

    lg = grid.ghosted.l
    τl = grid.ghosted.τl
    ∆τ = grid.ghosted.∆τ

    boundft = mdl.boundft
    gₑ = ft2gt.(EE,boundft)
    gₘ = ft2gt.(HH,boundft)

    εₜarr = mdl.εₜarr
    µₜarr = mdl.μₜarr
    µₗarr = mdl.μₗarr

    oind2shp = mdl.oind2shp
    oind2εind = mdl.oind2εind
    oind2μind = mdl.oind2μind
    εind2ε = mdl.εind2ε
    μind2μ = mdl.μind2μ

    cmpₑₜ = mdl.cmpₑₜ
    cmpₘₜ = mdl.cmpₘₜ
    cmpₘₗ = mdl.cmpₘₗ

    εind2εₜ = sub_pind2matprm(εind2ε, cmpₑₜ)
    μind2μₜ = sub_pind2matprm(μind2μ, cmpₘₜ)
    μind2μₗ = sub_pind2matprm(μind2μ, cmpₘₗ)

    iseₜ˔shp = mdl.iseₜ˔shp
    ishₜ˔shp = mdl.ishₜ˔shp
    ishₗ˔shp = true

    # Create temporary storages.
    # Because Ey and Hx are at the same locations, smoothing them can use the same object
    # index.
    pₜ_oind2d = create_oind_array(N)  # for εₜ (= εyy) and μₜ (= μxx)
    pₗ_oind2d = create_oind_array(N)  # for μₗ (= μzz)
    pₗ_oind2d′ = create_oind_array(N)  # for @assert below

    # Assign the material parameters.
    # Note that μₜarr uses gₑ rather than gₘ, because it is rank-0 tensor and its locations
    # are the E-field locations; see the definition MaxwellBase/assign_param!().
    # (See the figure in L10 - Eigenmode Analysis > 1.3 Waveguide mode analysis > Matrix
    # equation formulation; take the cross section through Ey, Hx, Hz.)
    assign_param!(εₜarr, tuple(pₗ_oind2d), oind2shp, oind2εind, εind2εₜ, gₑ, τl, isbloch)  # εₜ tensors (rank-0, so scalars)
    assign_param!(µₜarr, tuple(pₗ_oind2d′), oind2shp, oind2μind, μind2μₜ, gₑ, τl, isbloch)  # μₜ tensors (rank-0, so scalars)
    assign_param!(µₗarr, tuple(pₜ_oind2d), oind2shp, oind2μind, μind2μₗ, gₘ, τl, isbloch)  # μₗ tensors (rank-0, so scalars)

    # Some temporary arrays should be identical.  Specifically, εyy_oind2d and μxx_oind2d
    # are evaluated at the locations surrounding the εyy and μxx locations, which are the
    # same locations.  (See the figure in L10 - Eigenmode Analysis > 1.3 Waveguide mode
    # analysis > Matrix equation formulation; take the cross section through Ey, Hx, Hz.)
    # Therefore, εyy_oind2d and μxx_oind2d are evaluated at the same locations, so they
    # should be the same object indices.
    @assert isequal(pₗ_oind2d′, pₗ_oind2d)

    # Smooth the material parameters.
    smooth_param!(εₜarr, tuple(pₜ_oind2d), oind2shp, oind2εind, εind2εₜ, gₑ, l, lg, σ, ∆τ, iseₜ˔shp)  # εₜ tensors (rank-0, so scalars)
    smooth_param!(µₜarr, tuple(pₜ_oind2d), oind2shp, oind2μind, μind2μₜ, gₑ, l, lg, σ, ∆τ, ishₜ˔shp)  # μₜ tensors (rank-0, so scalars)
    smooth_param!(µₗarr, tuple(pₗ_oind2d), oind2shp, oind2μind, μind2μₗ, gₘ, l, lg, σ, ∆τ, ishₗ˔shp)  # μₗ tensors (rank-0, so scalars)

    return nothing
end

function complete_fields(β::Number,  # propagation constant
                         fₜ::AbsVecNumber,  # calculated transverse field Fₜ as column vector
                         ft::FieldType,  # type of input field Fₜ
                         ω::Number,
                         Ps::Tuple22{AbsMatNumber},
                         ∇̽s::Tuple22{AbsMatNumber},
                         πcmps::Tuple2{AbsMatNumber},
                         mdl::ModelTE)
    # Calculate the transverse components of the complementary field.
    βF′ₜgen = create_βF′ₜgen(ft, ω, Ps, ∇̽s, πcmps)
    f′ₜ = (βF′ₜgen * fₜ) ./ β

    eₜ, hₜ = ft==EE ? (fₜ,f′ₜ) : (f′ₜ,fₜ)

    # Calculate the longitudinal component of the H-field.
    iHₗgen = create_iHₗgen(ω, Ps, ∇̽s)
    hₗ = (iHₗgen * eₜ) ./ im

    # Take the Cartesian components of the fields and reshape them into arrays.
    sz_grid = size(mdl.grid)
    Ey = reshape(eₜ, sz_grid)
    Hx = reshape(hₜ, sz_grid)
    Hz = reshape(hₗ, sz_grid)

    return Ey, (Hx, Hz)
end
