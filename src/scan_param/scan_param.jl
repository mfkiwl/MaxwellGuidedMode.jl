export ParametrizedMode
export scan_param!

struct ParametrizedMode{K,Kₑ,Kₘ,Kθ,Kθₛ,NT<:NamedTuple}
    θ::NT  # named tuple of parameters to scan; entries should be vectors of same length; must have θ.ω
    β::ArrComplexF{Kθ}  # β(θ)
    E::NTuple{Kₑ,ArrComplexF{Kθₛ}}  # E(θ)
    H::NTuple{Kₘ,ArrComplexF{Kθₛ}}  # H(θ)
end

function ParametrizedMode(θ::NamedTuple, mdl::Model{K,Kₑ,Kₘ}) where {K,Kₑ,Kₘ}
    haskey(θ, :ω) || @error "θ = $θ should have entry named ω."
    all(issorted.(values(θ))) || @error "Each entry of θ should be sorted in ascending order for interpolation."

    szθ = length.(values(θ))
    Kθ = length(szθ)

    szₛ = size(mdl.grid)  # size of shape dimension
    Kθₛ = Kθ + length(szₛ)

    β = ArrComplexF(undef, szθ)
    E = ntuple(k->ArrComplexF(undef, szθ..., szₛ...), Val(Kₑ))
    H = ntuple(k->ArrComplexF(undef, szθ..., szₛ...), Val(Kₘ))

    return ParametrizedMode{K,Kₑ,Kₘ,Kθ,Kθₛ,typeof(θ)}(θ,β,E,H)
end

Base.length(pm::ParametrizedMode) = length(pm.θ.ω)

function scan_param!(pm::ParametrizedMode{K,Kₑ,Kₘ,Kθ},
                     βguessₑ::Number,  # guess β for ending parameters (e.g., largest ω = shortest λ) in pm
                     mdl::Model{K,Kₑ,Kₘ};
                     update_model!::Any=(mdl,θ)->nothing  # default: do not update model
                     ) where {K,Kₑ,Kₘ,Kθ}
    println("Begin parameter scan.")

    θ = pm.θ
    θkeys = keys(θ)
    θvals = values(θ)
    szθ = length.(θvals)
    Nθ = prod(szθ)

    CI = reverse(CartesianIndices(szθ))  # run from last corner of parameter space
    LI = LinearIndices(szθ)
    rng_shp = ntuple(k->Colon(), Val(K))  # represent all indices in shape dimensions
    t = @elapsed begin
        βguess = βguessₑ
        for χ = CI
            println("\tScanning $(Nθ+1-LI[χ]) out of $Nθ...")
            θval = t_ind(θvals, χ)
            θᵪ = NamedTuple{θkeys}(θval)
            ωᵪ = θᵪ.ω

            clear_objs!(mdl)
            update_model!(mdl, θᵪ)

            β, E, H = calc_mode(mdl, ωᵪ, βguess)

            pm.β[χ] = β  # filling from last corner of parameter space
            for k = 1:Kₑ
                view(pm.E[k], χ, rng_shp...) .= E[k]
            end
            for k = 1:Kₘ
                view(pm.H[k], χ, rng_shp...) .= H[k]
            end

            λ = LI[χ]  # linear index corresponding to χ
            if λ ≠ 1  # not first parameter (that is examined last)
                χnext = CI[λ-1]  # next Cartesian index
                ∆χ = χnext - χ
                sc = .!iszero.(∆χ.I)  # sc[k] = true if kth subscript will change
                if sum(sc) ≠ 1  # two or more subscripts will change
                    sc_max = findlast(sc)  # last dimension of changing subscript
                    ∆χclose = CartesianIndex(ntuple(k->(k==sc_max), Val(Kθ)))
                    χclose = χ + ∆χclose
                    βguess = β[χclose]
                else
                    βguess = β
                end
            end
        end
    end
    println("End parameter scan: $t seconds taken.")

    return nothing
end
