export set_ωpml!, set_boundft!, set_Npml!, set_kbloch!  # basic setter functions
export create_e⁻ⁱᵏᴸ
export clear_objs!  # plural because clearing both electric and magnetic quantities
export complete_fields

# Do not export Model; quality it with the package name MaxwellFDFD, because I would have
# similar types in other packages such as MaxwellSALT and MaxwellGuide.
Base.@kwdef mutable struct Model{K,Kₑ,Kₘ,
                                 Kₑₜ,Kₘₜ,Kₑₗ,Kₘₗ,
                                 K²,
                                 K₊₂,AK₊₂<:AbsArrComplexF{K₊₂}}
    # Frequency
    ωpml::Number = 0.0  # can be complex

    # Field type of equation
    ft_eq::FieldType

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

# Implement the following functions for each specific Model.
function calc_matparams! end
function complete_fields end

function field2vec(E::NTuple{Kₑ,AbsVecNumber{K}}, H::NTuple{Kₘ,AbsVecNumber{K}},
                   mdl::Model{K,Kₑ,Kₘ}
                   ) where {K,Kₑ,Kₘ}
    U = mdl.ft_eq==EE ? E : H
    cmpᵤₜ = mdl.ft_eq==EE ? mdl.cmpₑₜ : mdl.cmpₘₜ
    Uₜ = cat(U[cmpᵤₜ]..., dims=K+1)

    order_cmpfirst = mdl.order_cmpfirst
    if order_cmpfirst
        Uₜ = permutedims(Uₜ, circshift(1:K, 1))
    end

    return field_arr2vec(Uₜ; order_cmpfirst)
end

include("poynting.jl")
include("calc_mode.jl")

# Specific models
include("full.jl")
include("te.jl")
include("tm.jl")
