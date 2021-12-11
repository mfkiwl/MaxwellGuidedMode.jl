export create_paramops, create_curls, create_πcmps
export create_iF′ₗgen, create_iHₗgen, create_iEₗgen
export create_βF′ₜgen, create_βHₜgen, create_βEₜgen
export create_A
export calc_mode

function create_stretched_∆ls(mdl::Model)
    ωpml = mdl.ωpml
    grid = mdl.grid
    Npml = mdl.Npml
    s∆l = create_stretched_∆l(ωpml, grid, Npml)
    s∆l⁻¹ = invert_∆l(s∆l)

    boundft = mdl.boundft
    gtₑ = ft2gt.(EE, boundft)
    gtₘ = ft2gt.(HH, boundft)

    s∆lₑ = t_ind(s∆l, gtₑ)  # stretched ∆l's centered at E-field plane locations
    s∆lₘ = t_ind(s∆l, gtₘ)  # stretched ∆l's centered at H-field plane locations
    s∆lₑ⁻¹ = t_ind(s∆l⁻¹, gtₑ)
    s∆lₘ⁻¹ = t_ind(s∆l⁻¹, gtₘ)

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

function calc_mode(mdl::Model, ω::Real, βguess::Number)
    ## Create the eigenequation and solve it.
    Ps = create_paramops(mdl)
    ∇̽s = create_curls(mdl)
    πcmps = create_πcmps(mdl)

    ft_eq = mdl.ft_eq
    A = create_A(ft_eq, ω, Ps, ∇̽s, πcmps)

    nev = 1
    β², f = eigs(A; nev, sigma=βguess^2)

    fₜ = f[:,nev]
    β = .√β²[nev]

    E, H = complete_fields(β, fₜ, ft_eq, ω, Ps, ∇̽s, πcmps, mdl)
    normalize!(E, H, mdl)

    return (β=β, E=E, H=H)
end
