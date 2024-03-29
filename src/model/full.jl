# Alias of Model{K,Kₑ,Kₘ, Kₑₜ,Kₘₜ, Kₑₗ,Kₘₗ, K², K₊₂,AK₊₂} for the waveguide mode
# equations for full 2D cross section.
#
# The type parameters AK₊₁ and AK₊₂ specify device-specific arrays types (e.g., CuArray) and
# are user-defined in the constructor.
const ModelFull{AK₊₂} = Model{2,3,3, 2,2, 1,1, 4, 4,AK₊₂}

# Convenience constructor
function ModelFull(grid::Grid; Atype::Type=Array)
    ft_eq = EE

    cmpₛ = SInt(1,2)  # shapes in transverse dimension (xy-plane)

    cmpₑₜ = SInt(1,2)  # transverse E-field (x- and y-components)
    cmpₘₜ = SInt(1,2)  # transverse H-field (x- and y-components)

    cmpₑₗ = SInt(3)  # longitudinal E-field (z-component)
    cmpₘₗ = SInt(3)  # longitudinal H-field (z-component)

    iseₜ˔shp = false
    ishₜ˔shp = false

    return ModelFull{Atype{ComplexF,4}}(;ft_eq, grid, cmpₛ, cmpₑₜ, cmpₘₜ, cmpₑₗ, cmpₘₗ, iseₜ˔shp, ishₜ˔shp)
end

# Assign the material parameters on the grid and smooth them.
function calc_matparams!(mdl::ModelFull)
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
    µₜarr = mdl.μₜarr
    εₗarr = mdl.εₗarr
    µₗarr = mdl.μₗarr

    oind2shp = mdl.oind2shp
    oind2εind = mdl.oind2εind
    oind2μind = mdl.oind2μind
    εind2ε = mdl.εind2ε
    μind2μ = mdl.μind2μ

    cmpₑₜ = mdl.cmpₑₜ
    cmpₘₜ = mdl.cmpₘₜ
    cmpₑₗ = mdl.cmpₑₗ
    cmpₘₗ = mdl.cmpₘₗ

    εind2εₜ = sub_pind2matprm(εind2ε, cmpₑₜ)
    εind2εₗ = sub_pind2matprm(εind2ε, cmpₑₗ)
    μind2μₜ = sub_pind2matprm(μind2μ, cmpₘₜ)
    μind2μₗ = sub_pind2matprm(μind2μ, cmpₘₗ)

    iseₜ˔shp = mdl.iseₜ˔shp
    ishₜ˔shp = mdl.ishₜ˔shp
    iseₗ˔shp = true
    ishₗ˔shp = true

    # Create temporary storages.
    εxx_oind2d = create_oind_array(N)
    εyy_oind2d = create_oind_array(N)
    εzz_oind2d = create_oind_array(N)
    εoo_oind2d = create_oind_array(N)

    μxx_oind2d = create_oind_array(N)
    μyy_oind2d = create_oind_array(N)
    μzz_oind2d = create_oind_array(N)
    μoo_oind2d = create_oind_array(N)

    # Assign the material parameters.
    # (See the figure in L10 - Eigenmode Analysis > 1.3 Waveguide mode analysis > Matrix
    # equation formulation.)
    assign_param!(εₜarr, (εyy_oind2d,εxx_oind2d), oind2shp, oind2εind, εind2εₜ, gtₑ, τl, isbloch)  # diagonal entries of εₜ tensors
    assign_param!(εₜarr, tuple(μzz_oind2d), oind2shp, oind2εind, εind2εₜ, gtₑ, τl, isbloch)  # off-diagonal entries of εₜ tensors
    assign_param!(µₗarr, tuple(εoo_oind2d), oind2shp, oind2μind, μind2μₗ, gtₘ, τl, isbloch)  # μₗ tensors (rank-0, so scalars)

    assign_param!(µₜarr, (μyy_oind2d,μxx_oind2d), oind2shp, oind2μind, μind2μₜ, gtₘ, τl, isbloch)  # diagonal entries of μₜ tensors
    assign_param!(µₜarr, tuple(εzz_oind2d), oind2shp, oind2μind, μind2μₜ, gtₘ, τl, isbloch)  # off-diagonal entries of μₜ tensors
    assign_param!(εₗarr, tuple(μoo_oind2d), oind2shp, oind2εind, εind2εₗ, gtₑ, τl, isbloch)  # εₗ tensors (rank-0, so scalars)

    # Some temporary arrays should be identical.  For example, εyy_oind2d and μxx_oind2d are
    # evaluated at the locations surrounding the εyy and μxx locations, which are the same
    # locations.  (See the figure in L10 - Eigenmode Analysis > 1.3 Waveguide mode analysis
    # > Matrix equation formulation.)  Therefore, εyy_oind2d and μxx_oind2d are evaluated at
    # the same locations, so they should be the same object indices.
    @assert isequal(εxx_oind2d, μyy_oind2d)
    @assert isequal(εyy_oind2d, μxx_oind2d)
    @assert isequal(εoo_oind2d, εzz_oind2d)
    @assert isequal(μoo_oind2d, μzz_oind2d)

    # Smooth the material parameters.
    smooth_param!(εₜarr, (εxx_oind2d,εyy_oind2d), oind2shp, oind2εind, εind2εₜ, gtₑ, l, lg, σ, ∆τ, iseₜ˔shp)  # diagonal entries of εₜ tensors
    smooth_param!(εₜarr, tuple(εoo_oind2d), oind2shp, oind2εind, εind2εₜ, gtₑ, l, lg, σ, ∆τ, iseₜ˔shp)  # off-diagonal entries of εₜ tensors
    smooth_param!(µₗarr, tuple(μzz_oind2d), oind2shp, oind2μind, μind2μₗ, gtₘ, l, lg, σ, ∆τ, ishₗ˔shp)  # μₗ tensors (rank-0, so scalars)

    smooth_param!(µₜarr, (μxx_oind2d,μyy_oind2d), oind2shp, oind2μind, μind2μₜ, gtₘ, l, lg, σ, ∆τ, ishₜ˔shp)  # diagonal entries of μₜ tensors
    smooth_param!(µₜarr, tuple(μoo_oind2d), oind2shp, oind2μind, μind2μₜ, gtₘ, l, lg, σ, ∆τ, ishₜ˔shp)  # off-diagonal entries of μₜ tensors
    smooth_param!(εₗarr, tuple(εzz_oind2d), oind2shp, oind2εind, εind2εₗ, gtₑ, l, lg, σ, ∆τ, iseₗ˔shp)  # εₗ tensors (rank-0, so scalars)

    return nothing
end

function complete_fields(β::Number,  # propagation constant
                         fₜ::AbsVecNumber,  # calculated transverse field Fₜ as column vector
                         ft::FieldType,  # type of input field Fₜ
                         ω::Number,
                         Ps::Tuple22{AbsMatNumber},
                         ∇̽s::Tuple22{AbsMatNumber},
                         πcmps::Tuple2{AbsMatNumber},
                         mdl::ModelFull)
    ft′ = alter(ft)

    # Calculate the transverse components of the complementary field.
    βF′ₜgen = create_βF′ₜgen(ft, ω, Ps, ∇̽s, πcmps)
    f′ₜ = (βF′ₜgen * fₜ) ./ β

    # Calculate the longitudinal components of the main and complementary fields.
    iF′ₗgen = create_iF′ₗgen(ft, ω, Ps, ∇̽s)
    iFₗgen = create_iF′ₗgen(ft′, ω, Ps, ∇̽s)

    fₗ = (iFₗgen * f′ₜ) ./ im
    f′ₗ = (iF′ₗgen * fₜ) ./ im

    # Take the Cartesian components of the fields and reshape them into arrays.
    sz = size(mdl, ft)
    sz′ = size(mdl, ft′)
    @assert sz′ == sz

    fₜ = reshape(fₜ, sz)
    f′ₜ = reshape(f′ₜ, sz)

    order_cmpfirst = mdl.order_cmpfirst
    d = order_cmpfirst ? 1 : length(sz)

    Fx = selectdim(fₜ, d, 1)
    Fy = selectdim(fₜ, d, 2)

    F′x = selectdim(f′ₜ, d, 1)
    F′y = selectdim(f′ₜ, d, 2)

    sz_grid = size(mdl.grid)
    Fz = reshape(fₗ, sz_grid)
    F′z = reshape(f′ₗ, sz_grid)

    F = (Fx, Fy, Fz)
    F′ = (F′x, F′y, F′z)

    E, H = ft==EE ? (F,F′) : (F′,F)

    return (E=E, H=H)
end

# See the field configurations in L10 - Eigenmode Analysis > Matrix equation formulation.
function poynting(E::Tuple3{AbsMatNumber},  # (Ex, Ey, Ez)
                  H::Tuple3{AbsMatNumber},  # (Hx, Hy, Hz)
                  mdl::ModelFull)
    boundft = mdl.boundft
    grid = mdl.grid
    isbloch = grid.isbloch

    ∆l, ∆l⁻¹ = create_∆ls(grid, boundft)

    Ex, Ey, Ez = E
    Hx, Hy, Hz = H

    nX, nY = 1, 2

    Ftemp = ArrComplexF(undef, size(grid))  # matrix; temporary storage

    # Calculate Sx.
    Sx = ArrComplexF(undef, size(grid))  # matrix

    m̂xHz = Ftemp; interp_field!(m̂xHz, Hz, HH, nX, ∆l, ∆l⁻¹, boundft, isbloch)  # intepolate Hz at Ey-locations
    Sx .= Ey .* conj.(m̂xHz)  # EyHz evaluated at centers of x-normal edges

    m̂xHy = Ftemp; interp_field!(m̂xHy, Hy, HH, nX, ∆l, ∆l⁻¹, boundft, isbloch)  # intepolate Hy at Ez-locations
    m̂xHy .= Ez .* conj.(m̂xHy)  # EzHy evaluated at end points of x-normal edges; need to interpolate at centers
    Sx₋ = m̂xHy

    interp_field!(Sx, Sx₋, EE, nY, ∆l, ∆l⁻¹, boundft, isbloch, α=-1.0, overwrite=false)  # interpolate Sx₋ at centers of x-normal edges

    # Calculate Sy.
    Sy = ArrComplexF(undef, size(grid))  # matrix

    m̂yHz = Ftemp; interp_field!(m̂yHz, Hz, HH, nY, ∆l, ∆l⁻¹, boundft, isbloch)  # intepolate Hz at Ex-locations
    Sy .= -1.0 .* Ex .* conj.(m̂yHz)  # ExHz evaluated at centers of y-normal edges

    m̂yHx = Ftemp; interp_field!(m̂yHx, Hx, HH, nY, ∆l, ∆l⁻¹, boundft, isbloch)  # intepolate Hx at Ez-locations
    m̂yHx .= Ez .* conj.(m̂yHx)  # EzHx evaluated at end points of y-normal edges; need to interpolate at centers
    Sy₊ = m̂yHx

    interp_field!(Sy, Sy₊, EE, nX, ∆l, ∆l⁻¹, boundft, isbloch, overwrite=false)  # interpolate Sy₊ at centers of y-normal edges

    # Calculate Sz.
    Sz = ArrComplexF(undef, size(grid))  # matrix

    Sz₊ = Ftemp; Sz₊ .= Ex .* conj.(Hy)  # Ex and Hy are already co-located
    interp_field!(Sz, Sz₊, EE, nY, ∆l, ∆l⁻¹, boundft, isbloch)  # Sz₊ interpolated at centers of pixels

    Sz₋ = Ftemp; Sz₋ .= Ey .* conj.(Hx)  # Ey and Hx are already co-located
    interp_field!(Sz, Sz₋, EE, nX, ∆l, ∆l⁻¹, boundft, isbloch, α=-1.0, overwrite=false)  # Sz₋ interpolated at centers of pixels

    return Sx, Sy, Sz
end
