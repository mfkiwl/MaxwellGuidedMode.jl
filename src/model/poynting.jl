export poynting, powerₗ

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

# Used in poynting().
function interp_field!(G::AbsArrComplexF{K},  # where averaged fields are stored
                       F::AbsArrComplexF{K},  # field to average
                       ft::FieldType,  # type of field to average
                       nw::Int,  # direction of averaging
                       ∆l::Tuple2{NTuple{K,VecFloat}},
                       ∆l⁻¹::Tuple2{NTuple{K,VecFloat}},
                       boundft::SVec{K,FieldType},
                       isbloch::SBool{K};
                       kwargs...  # keyword arguments for apply_m̂!()
                       ) where {K}
    nft = Int(ft); nft′ = alter(nft)
    isfwd = boundft[nw]==ft
    apply_m̂!(G, F, nw, isfwd, ∆l[nft][nw], ∆l⁻¹[nft′][nw], isbloch[nw]; kwargs...)  # intepolate Hz at Ey-locations

    return nothing
end

# Calculate the power along the longitudinal direction.
function powerₗ(E::NTuple{Kₑ,AbsArrNumber{K}},
                H::NTuple{Kₘ,AbsArrNumber{K}},
                mdl::Model{K,Kₑ,Kₘ}
                ) where {K,Kₑ,Kₘ}
    S = poynting(E, H, mdl)
    Sz = S[end]

    gtₑ = ft2gt.(HH, mdl.boundft)
    lvxl = t_ind(mdl.grid.ghosted.l, gtₑ)  # locations of x-normal edges of E-field pixels surrounding Sz

    vc = VoxelwiseConstant(Sz, lvxl)
    Pz = integral(vc)

    return Pz
end

function LinearAlgebra.normalize!(E::NTuple{Kₑ,AbsArrNumber{K}},
                                  H::NTuple{Kₘ,AbsArrNumber{K}},
                                  mdl::Model{K,Kₑ,Kₘ}
                                  ) where {K,Kₑ,Kₘ}
    Pz = powerₗ(E, H, mdl)
    sqrtPz = √Pz

    for k = 1:Kₑ
        E[k] ./= sqrtPz
    end

    for k = 1:Kₘ
        H[k] ./= sqrtPz
    end

    return nothing
end
