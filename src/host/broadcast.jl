# broadcasting operations

export AbstractGPUArrayStyle

using Base.Broadcast

import Base.Broadcast: BroadcastStyle, Broadcasted, AbstractArrayStyle

"""
Abstract supertype for GPU array styles. The `N` parameter is the dimensionality.

Downstream implementations should provide a concrete array style type that inherits from
this supertype.
"""
abstract type AbstractGPUArrayStyle{N} <: AbstractArrayStyle{N} end

# Wrapper types otherwise forget that they are GPU compatible
# NOTE: don't directly use GPUArrayStyle here not to lose downstream customizations.
for (W, ctor) in Adapt.wrappers
  @eval begin
    BroadcastStyle(::Type{<:$W}) where {AT<:AbstractGPUArray} = BroadcastStyle(AT)
    backend(::Type{<:$W}) where {AT<:AbstractGPUArray} = backend(AT)
  end
end

# This Union is a hack. Ideally Base would have a Transpose <: WrappedArray <: AbstractArray
# and we could define our methods in terms of Union{AbstractGPUArray, WrappedArray{<:Any, <:AbstractGPUArray}}
@eval const GPUDestArray =
  Union{AbstractGPUArray, $((:($W where {AT <: AbstractGPUArray}) for (W, _) in Adapt.wrappers)...)}

# We purposefully only specialize `copyto!`, dependent packages need to make sure that they
# can handle:
# - `bc::Broadcast.Broadcasted{Style}`
# - `ex::Broadcast.Extruded`
# - `LinearAlgebra.Transpose{,<:AbstractGPUArray}` and `LinearAlgebra.Adjoint{,<:AbstractGPUArray}`, etc
#    as arguments to a kernel and that they do the right conversion.
#
# This Broadcast can be further customize by:
# - `Broadcast.preprocess(dest::AbstractGPUArray, bc::Broadcasted{Nothing})` which allows for a
#   complete transformation based on the output type just at the end of the pipeline.
# - `Broadcast.broadcasted(::Style, f)` selection of an implementation of `f` compatible
#   with `Style`
#
# For more information see the Base documentation.
@inline function Base.copyto!(dest::GPUDestArray, bc::Broadcasted{Nothing})
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    bc′ = Broadcast.preprocess(dest, bc)
    gpu_call(dest, bc′) do ctx, dest, bc′
        let I = CartesianIndex(@cartesianidx(dest))
            @inbounds dest[I] = bc′[I]
        end
        return
    end

    return dest
end

# Base defines this method as a performance optimization, but we don't know how to do
# `fill!` in general for all `GPUDestArray` so we just go straight to the fallback
@inline Base.copyto!(dest::GPUDestArray, bc::Broadcasted{<:Broadcast.AbstractArrayStyle{0}}) =
    copyto!(dest, convert(Broadcasted{Nothing}, bc))
