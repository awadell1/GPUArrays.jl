# common Base functionality
import Base: _RepeatInnerOuter

# Overload methods used by `Base.repeat`.
# No need to implement `repeat_inner_outer` since this is implemented in `Base` as
# `repeat_outer(repeat_inner(arr, inner), outer)`.
function _RepeatInnerOuter.repeat_inner(xs::AnyGPUArray{<:Any, N}, inner) where {N}
    return _repeat(xs; inner)
end

function _RepeatInnerOuter.repeat_outer(xs::AnyGPUArray{<:Any, N}, outer::NTuple{N}) where {N}
    return _repeat(xs; outer)
end

function _RepeatInnerOuter.repeat_inner_outer(xs::AnyGPUArray{<:Any, 1}, inner, outer)
    return _repeat(xs; inner, outer)
end

function _repeat(x::AbstractArray, counts::Integer...)
    N = max(ndims(x), length(counts))
    size_y = ntuple(d -> size(x,d) * get(counts, d, 1), N)
    size_x2 = ntuple(d -> isodd(d) ? size(x, 1+d÷2) : 1, 2*N)

    ## version without mutation
    # ignores = ntuple(d -> reshape(Base.OneTo(counts[d]), ntuple(_->1, 2d-1)..., :), length(counts))
    # y = reshape(broadcast(first∘tuple, reshape(x, size_x2), ignores...), size_y)

    # ## version with mutation
    size_y2 = ntuple(d -> isodd(d) ? size(x, 1+d÷2) : get(counts, d÷2, 1), 2*N)
    y = similar(x, size_y)
    reshape(y, size_y2) .= reshape(x, size_x2)
    y
end

function _repeat(x::AbstractArray; inner=1, outer=1)
    N = max(ndims(x), length(inner), length(outer))
    size_y = ntuple(d -> size(x, d) * get(inner, d, 1) * get(outer, d, 1), N)
    size_y3 = ntuple(3*N) do d3
        dim, class = divrem(d3+2, 3)  # e.g. for x::Matrix, [divrem(n+2,3) for n in 1:3*2]
        class == 0 && return get(inner, dim, 1)
        class == 1 && return size(x, dim)
        class == 2 && return get(outer, dim,1)
    end
    size_x3 = ntuple(3*N) do d3
        dim, class = divrem(d3+2, 3)
        class == 1 ? size(x, dim) : 1
    end
    y = similar(x, size_y)
    reshape(y, size_y3) .= reshape(x, size_x3)
    y
end

## PermutedDimsArrays

using Base: PermutedDimsArrays

# PermutedDimsArrays' custom copyto! doesn't know how to deal with GPU arrays
function PermutedDimsArrays._copy!(dest::PermutedDimsArray{T,N,<:Any,<:Any,<:AbstractGPUArray}, src) where {T,N}
    dest .= src
    dest
end

## concatenation

# hacky overloads to make simple vcat and hcat with numbers work as expected.
# we can't really make this work in general without Base providing
# a dispatch mechanism for output container type.
@inline Base.vcat(a::Number, b::AbstractGPUArray) =
    vcat(fill!(similar(b, typeof(a), (1,size(b)[2:end]...)), a), b)
@inline Base.hcat(a::Number, b::AbstractGPUArray) =
    hcat(fill!(similar(b, typeof(a), (size(b)[1:end-1]...,1)), a), b)
