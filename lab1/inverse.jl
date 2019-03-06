Lij(LU::AbstractArray{T, 2}, i::Int, j::Int) where T = i < j ? zero(T) : LU[i, j]::T
function Lij(LU::M, i::Int) where M
    n = size(LU, 1)
    Lij(LU, rem(i-1, n) + 1, div(i-1, n) + 1)
end

Uij(LU::AbstractArray{T, 2}, i::Int, j::Int) where T = i > j ? zero(T) : i == j ? one(T) : LU[i, j]::T
function Uij(LU::M, i::Int) where M
    n = size(LU, 1)
    Uij(LU, rem(i-1, n) + 1, div(i-1, n) + 1)
end

function LUDecomp(A::AbstractArray{T, 2}) where T
    LU = LUDecomp_core(A)
    getL(LU), getU(LU)
end

function LUDecomp_core(A::AbstractArray{T, 2}) where T
    LU = copy(A)
    n = size(A, 1)
    for i = 2:n
        @inbounds LU[1, i] = A[1, i] / A[1, 1]
    end

    for j = 2:n
        for i = j:n
            for k = 1:j-1
                @inbounds LU[i, j] -= Lij(LU, i, k) * Uij(LU, k, j)
            end
        end

        for i = j+1:n
            for k = 1:j-1
                @inbounds LU[j, i] -= Lij(LU, j, k) * Uij(LU, k, i)
            end
            @inbounds LU[j, i] /= LU[j, j]
        end
    end
    LU
end

function getL(LU::AbstractArray{T, 2}) where T
    L = zeros(size(LU))
    n = size(LU, 1)
    for ind = 1:length(L)
        if rem(ind-1, n) >= div(ind-1, n)
            @inbounds L[ind] = LU[ind]
        end
    end
    L
end

function getU(LU::AbstractArray{T, 2}) where T
    U = one(LU)
    n = size(LU, 1)
    for ind = 1:length(U)
        if rem(ind-1, n) < div(ind-1, n)
            @inbounds U[ind] = LU[ind]
        end
    end
    U
end

function inverse_L(LU::AbstractArray{T, 2}) where T
    Linv = one(LU)
    n = size(LU, 1)
    for j = 1:n
        @inbounds Linv[j, j] /= LU[j, j]
        for i = j+1:n
            for k = j:i-1
                @inbounds Linv[i, j] -= LU[i, k] * Linv[k, j]
            end
            @inbounds Linv[i, j] /= LU[i, i]
        end
    end
    Linv
end

function inverse_U(LU::AbstractArray{T, 2}) where T
    Uinv = one(LU)
    n = size(LU, 1)
    for j = 1:n
        for i = j+1:n
            for k = j:i-1
                @inbounds Uinv[j, i] -= Uinv[j, k] * LU[k, i]
            end
        end
    end
    Uinv
end

function inverse(A::AbstractArray{T, 2}) where T
    LU = LUDecomp_core(A)
    inverse_U(LU) * inverse_L(LU)
end
