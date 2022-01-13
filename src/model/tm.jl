# Alias of Model{K,Kₑ,Kₘ, Kₑₜ,Kₘₜ, Kₑₗ,Kₘₗ, K², K₊₂,AK₊₂} for the TM waveguide mode
# mode equations for 1D cross section.
#
# The type parameter AK₊₂ specify device-specific arrays types (e.g., CuArray) and is
# user-defined in the constructor.
#
# Note that the TM mode has (Ex, Ez, Hy) as nonzero components; this is unlike TM Maxwell's
# equations that have two H-field components and one E-field component.

const ModelTM{AK₊₂} = Model{1,2,1, 1,1, 1,0, 1, 3,AK₊₂}

# Convenience constructor
function ModelTM(grid::Grid; Atype::Type=Array)
    ft_eq = HH

    cmpₛ = SInt(1)  # shapes in transverse dimension (x-axis)

    cmpₑₜ = SInt(1)  # transverse E-field (x-component)
    cmpₘₜ = SInt(2)  # transverse H-field (y-component)

    cmpₑₗ = SInt(3)  # longitudinal E-field (z-component)
    cmpₘₗ = SInt()  # longitudinal H-field (no component)

    iseₜ˔shp = false
    ishₜ˔shp = true

    return ModelTM{Atype{ComplexF,3}}(;ft_eq, grid, cmpₛ, cmpₑₜ, cmpₘₜ, cmpₑₗ, cmpₘₗ, iseₜ˔shp, ishₜ˔shp)
end

# Assign the material parameters on the grid and smooth them.
function calc_matparams!(mdl::ModelTM)
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
    gtₑ = ft2gt.(EE,boundft)
    gtₘ = ft2gt.(HH,boundft)

    εₜarr = mdl.εₜarr
    εₗarr = mdl.εₗarr
    µₜarr = mdl.μₜarr

    oind2shp = mdl.oind2shp
    oind2εind = mdl.oind2εind
    oind2μind = mdl.oind2μind
    εind2ε = mdl.εind2ε
    μind2μ = mdl.μind2μ

    cmpₑₜ = mdl.cmpₑₜ
    cmpₘₜ = mdl.cmpₘₜ
    cmpₑₗ = mdl.cmpₑₗ

    εind2εₜ = sub_pind2matprm(εind2ε, cmpₑₜ)
    μind2μₜ = sub_pind2matprm(μind2μ, cmpₘₜ)
    εind2εₗ = sub_pind2matprm(εind2ε, cmpₑₗ)

    iseₜ˔shp = mdl.iseₜ˔shp
    ishₜ˔shp = mdl.ishₜ˔shp
    iseₗ˔shp = true

    # Create temporary storages.
    # Because Ex and Hy are at the same locations, smoothing them can use the same object
    # index.
    pₜ_oind2d = create_oind_array(N)  # for εₜ (= εxx) and μₜ (= μyy)
    pₗ_oind2d = create_oind_array(N)  # for εₗ (= εzz)
    pₗ_oind2d′ = create_oind_array(N)  # for @assert below

    # Assign the material parameters.
    # Note that εₜarr uses gtₘ rather than gtₑ, because it is rank-0 tensor and its locations
    # are the H-field locations; see the definition MaxwellBase/assign_param!().
    # (See the figure in L10 - Eigenmode Analysis > 1.3 Waveguide mode analysis > Matrix
    # equation formulation; take the cross section through Ez, Ex, Hy.)
    assign_param!(µₜarr, tuple(pₗ_oind2d), oind2shp, oind2μind, μind2μₜ, gtₘ, τl, isbloch)  # μₜ tensors (rank-0, so scalars)
    assign_param!(εₜarr, tuple(pₗ_oind2d′), oind2shp, oind2εind, εind2εₜ, gtₘ, τl, isbloch)  # εₜ tensors (rank-0, so scalars)
    assign_param!(εₗarr, tuple(pₜ_oind2d), oind2shp, oind2εind, εind2εₗ, gtₑ, τl, isbloch)  # εₗ tensors (rank-0, so scalars)

    # Some temporary arrays should be identical.  Specifically, εxx_oind2d and μyy_oind2d
    # are evaluated at the locations surrounding the εxx and μyy locations, which are the
    # same locations.  (See the figure in L10 - Eigenmode Analysis > 1.3 Waveguide mode
    # analysis > Matrix equation formulation; take the cross section through Ez, Ex, Hy.)
    # Therefore, εxx_oind2d and μyy_oind2d are evaluated at the same locations, so they
    # should be the same object indices.
    @assert isequal(pₗ_oind2d′, pₗ_oind2d)

    # Smooth the material parameters.
    smooth_param!(µₜarr, tuple(pₜ_oind2d), oind2shp, oind2μind, μind2μₜ, gtₘ, l, lg, σ, ∆τ, ishₜ˔shp)  # μₜ tensors (rank-0, so scalars)
    smooth_param!(εₜarr, tuple(pₜ_oind2d), oind2shp, oind2εind, εind2εₜ, gtₘ, l, lg, σ, ∆τ, iseₜ˔shp)  # εₜ tensors (rank-0, so scalars)
    smooth_param!(εₗarr, tuple(pₗ_oind2d), oind2shp, oind2εind, εind2εₗ, gtₑ, l, lg, σ, ∆τ, iseₗ˔shp)  # εₗ tensors (rank-0, so scalars)

    return nothing
end

function complete_fields(β::Number,  # propagation constant
                         fₜ::AbsVecNumber,  # calculated transverse field Fₜ as column vector
                         ft::FieldType,  # type of input field Fₜ
                         ω::Number,
                         Ps::Tuple22{AbsMatNumber},
                         ∇̽s::Tuple22{AbsMatNumber},
                         πcmps::Tuple2{AbsMatNumber},
                         mdl::ModelTM)
    # Calculate the transverse components of the complementary field.
    βF′ₜgen = create_βF′ₜgen(ft, ω, Ps, ∇̽s, πcmps)
    f′ₜ = (βF′ₜgen * fₜ) ./ β

    eₜ, hₜ = ft==EE ? (fₜ,f′ₜ) : (f′ₜ,fₜ)

    # Calculate the longitudinal component of the E-field.
    iEₗgen = create_iEₗgen(ω, Ps, ∇̽s)
    eₗ = (iEₗgen * hₜ) ./ im

    # Take the Cartesian components of the fields and reshape them into arrays.
    sz_grid = size(mdl.grid)
    Ex = reshape(eₜ, sz_grid)
    Ez = reshape(eₗ, sz_grid)
    Hy = reshape(hₜ, sz_grid)

    return (E=(Ex, Ez), H=Hy)
end

function poynting(E::Tuple2{AbsVecNumber},  # (Ex, Ez)
                  H::AbsVecNumber,  # Hy
                  mdl::ModelTM)
    boundft = mdl.boundft
    grid = mdl.grid
    isbloch = grid.isbloch

    ∆l, ∆l⁻¹ = create_∆ls(grid, boundft)

    Ex, Ez = E
    Hy = H

    nX = 1

    # Calculate Sx.
    m̂xHy = VecComplexF(undef, size(grid))
    isfwd = boundft[nX]==HH
    apply_m̂!(m̂xHy, Hy, nX, isfwd, ∆lₘ[nX], ∆lₑ⁻¹[nX], isbloch[nX])
    Sx = -0.5real.(Ez .* conj.(m̂xHy))

    # Calculate Sz.
    Sz = 0.5real.(Ex .* conj.(Hy))  # real array located at Ex

    return Sx, Sz
end
