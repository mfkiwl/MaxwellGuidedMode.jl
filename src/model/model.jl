export set_ωpml!, set_boundft!, set_Npml!, set_kbloch!  # basic setter functions
export create_e⁻ⁱᵏᴸ
export clear_objs!  # plural because clearing both electric and magnetic quantities
export create_paramops, create_curls, create_πcmps
export create_iF′ₗgen, create_iHₗgen, create_iEₗgen
export create_βF′ₜgen, create_βHₜgen, create_βEₜgen
export create_A
export complete_fields
export poynting

# Do not export Model; quality it with the package name MaxwellFDFD, because I would have
# similar types in other packages such as MaxwellSALT and MaxwellGuide.
Base.@kwdef mutable struct Model{K,Kₑ,Kₘ,
                                 Kₑₜ,Kₘₜ,Kₑₗ,Kₘₗ,
                                 K²,
                                 K₊₂,AK₊₂<:AbsArrComplexF{K₊₂}}
    # Frequency
    ωpml::Number = 0.0  # can be complex

    # Grid
    grid::Grid{K}
    cmpₛ::SInt{K}  # Cartesian components of shape dimension
    cmpₑₜ::SInt{Kₑₜ}  # Cartesian components of transverse E-field
    cmpₘₜ::SInt{Kₘₜ}  # Cartesian components of transverse H-field
    cmpₑₗ::SInt{Kₑₗ}  # Cartesian components of longitudinal E-field
    cmpₘₗ::SInt{Kₘₗ}  # Cartesian components of longitudinal H-field

    # Boundary properties (in transverse directions)
    boundft::SVec{K,FieldType} = SVec(ntuple(k->EE, Val(K)))
    Npml::Tuple2{SInt{K}} = (SVec(ntuple(k->0, Val(K))), SVec(ntuple(k->0, Val(K))))
    kbloch::SFloat{K} = SVec(ntuple(k->0.0, Val(K)))  # [kx_bloch, ky_bloch, kz_bloch]

    # Material parameter arrays
    εₜarr::AK₊₂ = create_param_array(grid.N, ncmp=Kₑₜ)  # filled with zeros
    μₜarr::AK₊₂ = create_param_array(grid.N, ncmp=Kₘₜ)  # filled with zeros

    εₗarr::AK₊₂ = create_param_array(grid.N, ncmp=Kₑ-Kₑₜ)  # filled with zeros
    μₗarr::AK₊₂ = create_param_array(grid.N, ncmp=Kₘ-Kₘₜ)  # filled with zeros

    # Temporary storages for assignment and smoothing of material parameters
    oind2shp::Vector{Shape{K,K²}} = Shape{K,K²}[]
    oind2εind::Vector{ParamInd} = ParamInd[]
    oind2μind::Vector{ParamInd} = ParamInd[]
    εind2ε::Vector{S²ComplexF3} = S²ComplexF3[]
    μind2μ::Vector{S²ComplexF3} = S²ComplexF3[]

    # Boolean flags indicating if the transverse E- and H-field dimensions are orthogonal to
    # the shape dimensions.  Used for material parameter smoothing in calc_matparams!().
    iseₜ˔shp::Bool
    ishₜ˔shp::Bool

    # Indexing scheme for DOFs
    order_cmpfirst::Bool = true
end

function Base.size(mdl::Model{K,Kₑ,Kₘ,Kₑₜ,Kₘₜ}, ft::FieldType) where {K,Kₑ,Kₘ,Kₑₜ,Kₘₜ}
    Kf = ft==EE ? Kₑₜ : Kₘₜ
    sz_grid = size(mdl.grid)
    sz = mdl.order_cmpfirst ? (Kf, sz_grid...) : (sz_grid..., Kf)

    return sz
end

Base.length(mdl::Model, ft::FieldType) = prod(size(mdl, ft))

# Basic setters
set_ωpml!(mdl::Model, ωpml::Number) = (mdl.ωpml = ωpml; nothing)
set_boundft!(mdl::Model{K}, boundft::AbsVec{FieldType}) where {K} = (mdl.boundft = SVec{K}(boundft); nothing)
set_Npml!(mdl::Model{K}, Npml::Tuple2{AbsVecInteger}) where {K} = (mdl.Npml = SVec{K}.(Npml); nothing)
set_kbloch!(mdl::Model{K}, kbloch::AbsVecReal) where {K} = (mdl.kbloch = SVec{K}(kbloch); nothing)

create_e⁻ⁱᵏᴸ(mdl::Model) = exp.(-im .* mdl.kbloch .* mdl.grid.L)

# Main functions
function clear_objs!(mdl::Model)
    mdl.εₜarr .= 0
    mdl.μₜarr .= 0

    mdl.εₗarr .= 0
    mdl.μₗarr .= 0

    empty!(mdl.oind2shp)
    empty!(mdl.oind2εind)
    empty!(mdl.oind2μind)
    empty!(mdl.εind2ε)
    empty!(mdl.μind2μ)

    return nothing
end

MaxwellBase.add_obj!(mdl::Model{K}, matname::String, shapes::AbsVec{<:Shape{K}};
                     ε::MatParam=1.0, μ::MatParam=1.0) where {K} =
    add_obj!(mdl, matname, ε=ε, μ=μ, tuple(shapes...))

function MaxwellBase.add_obj!(mdl::Model{K}, matname::String, shapes::Shape{K}...;
                              ε::MatParam=1.0, μ::MatParam=1.0) where {K}
    mat = Material(matname, ε=ε, μ=μ)
    isbidirectional(EE, mat, mdl) || @error "ε = $ε should not couple transverse and longitudinal fields."
    isbidirectional(HH, mat, mdl) || @error "μ = $μ should not couple transverse and longitudinal fields."

    for shp = shapes  # shapes is tuple
        obj = Object(shp, mat)
        add_obj!(mdl.oind2shp, (mdl.oind2εind,mdl.oind2μind), (mdl.εind2ε,mdl.μind2μ), obj)
    end

    return nothing
end

# Returns true if the material parameter does not couple the transverse and longitudinal
# components of the fields; false otherwise.
function isbidirectional(ft::FieldType, mat::Material, mdl::Model)
    nft = Int(ft)
    cmpₜ = (mdl.cmpₑₜ, mdl.cmpₘₜ)[nft]
    cmpₗ = (mdl.cmpₑₗ, mdl.cmpₘₗ)[nft]

    param = mat.param[nft]

    return iszero(param[cmpₜ,cmpₗ]) && iszero(param[cmpₗ,cmpₜ])
end

function create_stretched_∆ls(mdl::Model)
    ωpml = mdl.ωpml
    grid = mdl.grid
    Npml = mdl.Npml
    s∆l = create_stretched_∆l(ωpml, grid, Npml)
    s∆l⁻¹ = invert_∆l(s∆l)

    boundft = mdl.boundft
    gₑ = ft2gt.(EE, boundft)
    gₘ = ft2gt.(HH, boundft)

    s∆lₑ = t_ind(s∆l, gₑ)  # stretched ∆l's centered at E-field plane locations
    s∆lₘ = t_ind(s∆l, gₘ)  # stretched ∆l's centered at H-field plane locations
    s∆lₑ⁻¹ = t_ind(s∆l⁻¹, gₑ)
    s∆lₘ⁻¹ = t_ind(s∆l⁻¹, gₘ)

    return s∆lₑ, s∆lₘ, s∆lₑ⁻¹, s∆lₘ⁻¹
end

function create_paramops(mdl::Model{K,Kₑ,Kₘ,Kₑₜ,Kₘₜ}) where {K,Kₑ,Kₘ,Kₑₜ,Kₘₜ}
    s∆lₑ, s∆lₘ, s∆lₑ⁻¹, s∆lₘ⁻¹ = create_stretched_∆ls(mdl)
    calc_matparams!(mdl)  # assignment and smoothing; implemented for each specialized alias of Model

    boundft = mdl.boundft
    isbloch = mdl.grid.isbloch
    e⁻ⁱᵏᴸ = create_e⁻ⁱᵏᴸ(mdl)

    isfwd_inₑ = boundft.!=EE
    isfwd_inₘ = boundft.!=HH

    Pεₜ = Kₑₜ==1 ? create_paramop(mdl.εₜarr; mdl.order_cmpfirst) :
                   create_paramop(mdl.εₜarr, isfwd_inₑ, s∆lₘ, s∆lₑ⁻¹, isbloch, e⁻ⁱᵏᴸ; mdl.order_cmpfirst)
    Pμₜ = Kₘₜ==1 ? create_paramop(mdl.μₜarr; mdl.order_cmpfirst) :
                   create_paramop(mdl.μₜarr, isfwd_inₘ, s∆lₑ, s∆lₘ⁻¹, isbloch, e⁻ⁱᵏᴸ; mdl.order_cmpfirst)

    # The following lines work even if the model does not deal with the corresponding
    # longitudinal components.
    Pεₗ = create_paramop(mdl.εₗarr; mdl.order_cmpfirst)  # 0×0 matrix for Kₑₗ = 0 (no longitudinal E-field)
    Pμₗ = create_paramop(mdl.μₗarr; mdl.order_cmpfirst)  # 0×0 matrix for Kₘₗ = 0 (no longitudinal H-field)

    return (Pεₜ,Pμₜ), (Pεₗ,Pμₗ)
end

function create_curls(mdl::Model)
    _, _, s∆lₑ⁻¹, s∆lₘ⁻¹ = create_stretched_∆ls(mdl)

    boundft = mdl.boundft

    cmpₛ = mdl.cmpₛ
    cmpₑₜ, cmpₘₜ = mdl.cmpₑₜ, mdl.cmpₘₜ
    cmpₑₗ, cmpₘₗ = mdl.cmpₑₗ, mdl.cmpₘₗ

    isbloch = mdl.grid.isbloch
    e⁻ⁱᵏᴸ = create_e⁻ⁱᵏᴸ(mdl)

    isfwdₑ = boundft.==EE
    isfwdₘ = boundft.==HH

    # The following lines work even if the model does not deal with the corresponding
    # longitudinal components.
    ∇̽ₑₜ = create_curl(isfwdₑ, s∆lₘ⁻¹, isbloch, e⁻ⁱᵏᴸ; cmp_shp=cmpₛ, cmp_out=cmpₘₗ, cmp_in=cmpₑₜ, mdl.order_cmpfirst)
    ∇̽ₘₜ = create_curl(isfwdₘ, s∆lₑ⁻¹, isbloch, e⁻ⁱᵏᴸ; cmp_shp=cmpₛ, cmp_out=cmpₑₗ, cmp_in=cmpₘₜ, mdl.order_cmpfirst)

    ∇̽ₑₗ = create_curl(isfwdₑ, s∆lₘ⁻¹, isbloch, e⁻ⁱᵏᴸ; cmp_shp=cmpₛ, cmp_out=cmpₘₜ, cmp_in=cmpₑₗ, mdl.order_cmpfirst)
    ∇̽ₘₗ = create_curl(isfwdₘ, s∆lₑ⁻¹, isbloch, e⁻ⁱᵏᴸ; cmp_shp=cmpₛ, cmp_out=cmpₑₜ, cmp_in=cmpₘₗ, mdl.order_cmpfirst)

    return (∇̽ₑₜ,∇̽ₘₜ), (∇̽ₑₗ,∇̽ₘₗ)
end

function create_πcmps(mdl::Model{K}) where {K}
    grid = mdl.grid
    cmpₑₜ, cmpₘₜ = mdl.cmpₑₜ, mdl.cmpₘₜ

    permute = reverse(SVec(ntuple(identity, Val(K))))
    scale₀ = SFloat(1,-1)

    πcmpₑ = create_πcmp(grid.N, permute, scale=scale₀[cmpₑₜ]; mdl.order_cmpfirst)
    πcmpₘ = create_πcmp(grid.N, permute, scale=scale₀[cmpₘₜ]; mdl.order_cmpfirst)

    return πcmpₑ, πcmpₘ
end

create_iHₗgen(ω, Ps::Tuple22{AbsMatNumber}, ∇̽s::Tuple22{AbsMatNumber}) = create_iF′ₗgen(EE, ω, Ps, ∇̽s)
create_iEₗgen(ω, Ps::Tuple22{AbsMatNumber}, ∇̽s::Tuple22{AbsMatNumber}) = create_iF′ₗgen(HH, ω, Ps, ∇̽s)

create_iF′ₗgen(ft, ω, Ps::Tuple22{AbsMatNumber}, ∇̽s::Tuple22{AbsMatNumber}) =
    create_iF′ₗgen(ft, ω, Ps[2], ∇̽s[1])

# Create the operator that generates im * F′ₗ from Fₜ, where F′ is the field complementary
# to F (e.g., F′ = H-field for F = E-field).
function create_iF′ₗgen(ft::FieldType,  # type of input field Fₜ, not output field F′ₗ
                        ω::Number,
                        Pₗs::Tuple2{AbsMatNumber},
                        ∇̽ₜs::Tuple2{AbsMatNumber})
    nft = Int(ft)
    nft′ = alter(nft)

    α = ft==EE ? -1/ω : 1/ω

    P′ₗ = Pₗs[nft′]
    ∇̽ₜ = ∇̽ₜs[nft]

    L = α .* (P′ₗ \ ∇̽ₜ)

    return L
end


create_βHₜgen(ω, Ps, ∇̽s, πcmps) = create_βF′ₜgen(EE, ω, Ps, ∇̽s, πcmps)
create_βEₜgen(ω, Ps, ∇̽s, πcmps) = create_βF′ₜgen(HH, ω, Ps, ∇̽s, πcmps)

# Create the operator that generates β * F′ₜ, where F′ is the field complementary to F (e.g.,
# F′ = H-field for F = E-field).
create_βF′ₜgen(ft, ω, Ps, ∇̽s, πcmps) = sum(create_βF′ₜgen_comps(ft, ω, Ps, ∇̽s, πcmps))

function create_βF′ₜgen_comps(ft::FieldType,  # type of input field Fₜ, not output field F′ₗ
                              ω::Number,
                              Ps::Tuple22{AbsMatNumber},
                              ∇̽s::Tuple22{AbsMatNumber},
                              πcmps::Tuple2{AbsMatNumber})
    nft = Int(ft)
    nft′ = alter(nft)

    α = ft==EE ? ω : -ω

    Pₜs, Pₗs = Ps
    ∇̽ₜs, ∇̽ₗs = ∇̽s

    Pₜ = Pₜs[nft]
    ∇̽′ₗ = ∇̽ₗs[nft′]
    πcmp = πcmps[nft]

    βF′ₜgen₁ = (α .* πcmp) * Pₜ

    if length(Pₗs[nft′]) > 0
        iF′ₗgen = create_iF′ₗgen(ft, ω, Pₗs, ∇̽ₜs)
        βF′ₜgen₂ = (πcmp * ∇̽′ₗ) * iF′ₗgen
    else
        βF′ₜgen₂ = spzeros(eltype(βF′ₜgen₁), size(βF′ₜgen₁)...)
    end

    return βF′ₜgen₁, βF′ₜgen₂
end

function create_A(ft::FieldType,  # type of input field Fₜ
                  ω::Number,
                  Ps::Tuple22{AbsMatNumber},
                  ∇̽s::Tuple22{AbsMatNumber},
                  πcmps::Tuple2{AbsMatNumber})
    ft′ = alter(ft)

    # Old implementation.  Though more straightforward, this implementation does not
    # gurantee the vector calculus identity βFₜgen₂ * βF′ₜgen₂ = 0 due to rounding errors.
    #
    # βF′ₜgen = create_βF′ₜgen(ft, ω, Ps, ∇̽s, πcmps)
    # βFₜgen = create_βF′ₜgen(ft′, ω, Ps, ∇̽s, πcmps)
    # A = βFₜgen * βF′ₜgen

    βF′ₜgen₁, βF′ₜgen₂ = create_βF′ₜgen_comps(ft, ω, Ps, ∇̽s, πcmps)
    βFₜgen₁, βFₜgen₂ = create_βF′ₜgen_comps(ft′, ω, Ps, ∇̽s, πcmps)

    A = βFₜgen₁ * βF′ₜgen₁ + βFₜgen₁ * βF′ₜgen₂ + βFₜgen₂ * βF′ₜgen₁  # βFₜgen₂ * βF′ₜgen₂ = 0 always

    return A
end

# Implement the following functions for each specific Model.
function calc_matparams! end
function complete_fields end

# The function Re{⋅} is linear for real coefficients.  In other words, for complex numbers
# z = x + iy and w = u + iv,
#
# Re{p z + q w} = Re{p(x+iy) + q(u+iv)} = p x + q u = p Re{z} + q Re{w}.
#
# Therefore, when averaging the E- and H-fields for the Poynting vector calculation, the
# weight factors ∆l shouldn't contain the PML scale factors (i.e., we should not use s∆l).
# As long as we use real ∆l, we can average the Poynting vector before the last step, e.g.,
# Sz = ½ Re{Ex Hy＊ - Ey Hx＊} = ½ Re{Ex Hy＊} - ½ Re{Ey Hx＊}.
#
# Also note that the time-averaged Poynting vector is meaningful only for real frequency ω.
# This is usually not a problem because
# - in source-driven problems we use real frequency;
# - in waveguide mode problems we use ω-to-β formulation, where we use real ω and calculate
# complex β.
# However, note that in SALT we deal with complex ω below the lasing threshold.  Thankfully
# we don't need to calculate the Poynting vector for such nonlasing modes.
#
# I don't care about eliminating array allocations here, because the Poynting vector
# calculation is much cheaper than solution calculation.  If I really need to eliminate
# allocations, consider using for loops; otherwise such elimination will be difficult
# because of field averaging.
#
# For simplicity, always calculate the Poynting vectors as quantities entering the E-field
# voxels.
#
# On the Yee grid, the different Cartesian components of different vector fields are defined
# at unique locations.  Positioning these Cartesian components of vector fields at their
# correct locations (with the correct half-grid-point offsets) can be confusing, because
# there are quite a few different Yee grid types.  The 3D Yee grid for the 3D Maxwell
# equations is the most standard configuration, so the positions of the Cartesian components
# of the E- and H-fields in there are relatively well-known.  But in the grid configurations
# for other types of equations, such as the 2D TE and TM Maxwell equations and even the
# waveguide mode equation, the correct positions of the Cartesian components of the E- and H-
# fields can be confusing.  The correct positions of the Cartesian components of the
# Poyinting vector are even more confusing, because their calculation involves cross
# products and averaging.  So, the users must make sure to know where their vector fields
# are defined.  In the future, I may need to provide some wrapper functions that help the
# users on this regard.
function poynting end

include("full.jl")
include("te.jl")
include("tm.jl")
